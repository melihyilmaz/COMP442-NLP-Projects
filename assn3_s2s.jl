import Pkg; using Pkg; Pkg.add("Knet")

Pkg.add("IterTools"); Pkg.add("AutoGrad")

using Knet, Test, Base.Iterators, IterTools, Random # , LinearAlgebra, StatsBase
using AutoGrad: @gcheck  # to check gradients, use with Float64
#Knet.atype() = KnetArray{Float32}  # determines what Knet.param() uses.
macro size(z, s); esc(:(@assert (size($z) == $s) string(summary($z),!=,$s))); end # for debugging

struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    M = 100000
    wdict = Dict()
    wcount = Dict()
    w2i(x) = get!(wdict, x, 1+length(wdict))
    w2c(key) = haskey(wcount, key) ? wcount[key] = wcount[key] + 1 : get!(wcount, key, 1)
    wcount[unk] = M; wcount[eos] = M
    i2w = []; 

    
    for line in eachline(file)
        words = tokenizer(line)
        w2c.(words)
    end
    
    sortedcount = sort(collect(wcount), by=x->x[2])
    words = sortedcount[findfirst(x-> x[2]>=mincount, sortedcount):length(sortedcount)]
    
    #vocabsize excludes unk & eos
    if(length(words) > vocabsize)
        words = words[length(words) - vocabsize + 1 : length(words)]
    end

    map(x-> w2i(x[1]) , words)
    map(x-> push!(i2w, x[1]), words)
    
    Vocab(wdict, i2w, wdict[unk], wdict[eos], tokenizer)
end
#=
function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    M = 1000000000000000
    wdict = Dict()
    wcount = Dict()
    w2i(x) = get!(wdict, x, 1+length(wdict))
    w2c(key) = haskey(wcount, key) ? wcount[key] = wcount[key] + 1 : get!(wcount, key, 1)
    wcount[eos] = M; wcount[unk] = M - 1
    i2w = []; 

    
    for line in eachline(file)
        words = tokenizer(line)
        w2c.(words)
    end
    
    sortedcount = sort(collect(wcount), by=x->x[2])
    words = sortedcount[findfirst(x-> x[2]>=mincount, sortedcount):length(sortedcount)]
    
    #vocabsize excludes unk & eos
    if(length(words) > vocabsize)
        words = words[length(words) - vocabsize + 1 : length(words)]
    end
    
    words = reverse(words)

    map(x-> w2i(x[1]) , words)
    map(x-> push!(i2w, x[1]), words)
    
    Vocab(wdict, i2w, wdict[unk], wdict[eos], tokenizer)
end
=#

struct TextReader
    file::String
    vocab::Vocab
end

function Base.iterate(r::TextReader, s=nothing)
    w2i(x) = get(r.vocab.w2i, x, r.vocab.unk)
    if (s === nothing) 
        s = open(r.file, "r")
    end

    if eof(s) 
        close(s)
        return nothing
    
    else
        tmp = readline(s)
        line = r.vocab.tokenizer(tmp)
        words = w2i.(line) 
        return words, s
    end    
end

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    Embed(param(embedsize, vocabsize))
end

function (l::Embed)(x)
    embedsz, vocabsz = size(l.w)
    tmparr = [embedsz]
    for dim in size(x)
        push!(tmparr, dim)
    end
    reshape(l.w[:,collect(flatten(x))], tuple(tmparr...))
end

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    w = param(outputsize, inputsize)
    b = param0(outputsize)
    Linear(w,b)
end

function (l::Linear)(x)
    l.w * x .+ l.b
end

function mask!(a,pad)
    x,y = size(a)
    
    for i = 1:x
        tmp_mem = []
        isfirst = true
        for j = 1:y
            if a[i, j] == pad
                
                if isfirst
                    isfirst = false
                else
                    push!(tmp_mem, j)
                end
            else
                isfirst = true
                tmp_mem = []
            end
        end
        tmp_mem = convert(Array{Int,1}, tmp_mem)
        a[i, tmp_mem] .= 0
    end
    return a
end

datadir = "datasets/tr_to_en"

if !isdir(datadir)
    download("http://www.phontron.com/data/qi18naacl-dataset.tar.gz", "qi18naacl-dataset.tar.gz")
    run(`tar xzf qi18naacl-dataset.tar.gz`)
end


if !isdefined(Main, :tr_vocab)
    tr_vocab = Vocab("$datadir/tr.train", mincount=5)
    en_vocab = Vocab("$datadir/en.train", mincount=5)
    tr_train = TextReader("$datadir/tr.train", tr_vocab)
    en_train = TextReader("$datadir/en.train", en_vocab)
    tr_dev = TextReader("$datadir/tr.dev", tr_vocab)
    en_dev = TextReader("$datadir/en.dev", en_vocab)
    tr_test = TextReader("$datadir/tr.test", tr_vocab)
    en_test = TextReader("$datadir/en.test", en_vocab)
    @info "Testing data"
    @test length(tr_vocab.i2w) == 38126
    @test length(first(tr_test)) == 16
    @test length(collect(tr_test)) == 5029
end

struct MTData
    src::TextReader        # reader for source language data
    tgt::TextReader        # reader for target language data
    batchsize::Int         # desired batch size
    maxlength::Int         # skip if source sentence above maxlength
    batchmajor::Bool       # batch dims (B,T) if batchmajor=false (default) or (T,B) if true.
    bucketwidth::Int       # batch sentences with length within bucketwidth of each other
    buckets::Vector        # sentences collected in separate arrays called buckets for each length range
    batchmaker::Function   # function that turns a bucket into a batch.
end

#batchsize 128
function MTData(src::TextReader, tgt::TextReader; batchmaker = arraybatch, batchsize = 128, maxlength = typemax(Int),
                batchmajor = false, bucketwidth = 10, numbuckets = min(128, maxlength ÷ bucketwidth))
    buckets = [ [] for i in 1:numbuckets ] # buckets[i] is an array of sentence pairs with similar length
    MTData(src, tgt, batchsize, maxlength, batchmajor, bucketwidth, buckets, batchmaker)
end

Base.IteratorSize(::Type{MTData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{MTData}) = Base.HasEltype()
Base.eltype(::Type{MTData}) = NTuple{2}

function Base.iterate(d::MTData, state=nothing)
    if (state === nothing) 
        
        for i = 1:length(d.buckets)
            d.buckets[i] = []
        end
        src = d.src
        tgt = d.tgt
        src = Iterators.Stateful(src)
        tgt = Iterators.Stateful(tgt)
    else
        src = state[1]
        tgt = state[2]
    end
    
    
    if(isempty(src)&&isempty(tgt))
        for i = 1:length(d.buckets)
            if(length(d.buckets[i]) > 0)
                tmp_batch = d.batchmaker(d, d.buckets[i])
                 if(d.batchmajor == true)
                    tmp_batch = (transpose(tmp_batch[1]), transpose(tmp_batch[2]))
                end
                d.buckets[i] = []
                return (tmp_batch, (src, tgt))
            end
        end
    end    
        
    while(!isempty(src) && !isempty(tgt))
        sentences = (popfirst!(src), popfirst!(tgt))
        src_sentence = sentences[1]
        tgt_sentence = sentences[2]
        src_length = length(src_sentence)
        
        if(src_length > d.maxlength)
            continue
        elseif(length(d.buckets)*d.bucketwidth < src_length)
            index_in_buckets = length(d.buckets)
        else
            index_in_buckets = ceil(src_length/d.bucketwidth)
        end
        
        index_in_buckets = convert(Int64, index_in_buckets)
        push!(d.buckets[index_in_buckets], (src_sentence, tgt_sentence))
        
        if(isempty(src) && isempty(tgt))
                tmp_batch = d.batchmaker(d, d.buckets[index_in_buckets])
                if(d.batchmajor == true)
                    tmp_batch = (transpose(tmp_batch[1]), transpose(tmp_batch[2]))
                end
                d.buckets[index_in_buckets] = []
                return (tmp_batch, (src, tgt))
        end  
        
        if(length(d.buckets[index_in_buckets]) == d.batchsize)
            tmp_batch = d.batchmaker(d, d.buckets[index_in_buckets])
            if(d.batchmajor == true)
                tmp_batch = (transpose(tmp_batch[1]), transpose(tmp_batch[2]))
            end
            d.buckets[index_in_buckets] = []
            return (tmp_batch, (src, tgt))
        end 
    end   
end

function arraybatch(d::MTData, bucket)
    # Your code here
    x = []
    y = []
    
    padded_x = Array{Int64,1}[]
    padded_y = Array{Int64,1}[]
    
    max_length_x = 0
    max_length_y = 0
    
    for sent_pair in bucket
        push!(x, sent_pair[1])
        push!(sent_pair[2], d.tgt.vocab.eos)
        pushfirst!(sent_pair[2], d.tgt.vocab.eos)
        push!(y, sent_pair[2])
        
        if(length(sent_pair[1]) > max_length_x)
            max_length_x = length(sent_pair[1])
        end
        
        if(length(sent_pair[2]) > max_length_y)
            max_length_y = length(sent_pair[2])
        end
    end
    for sent_pair in zip(x,y)
        x_pad_length = max_length_x - length(sent_pair[1])
        y_pad_length = max_length_y - length(sent_pair[2])
        x_pad_seq = repeat([d.src.vocab.eos], x_pad_length)
        y_pad_seq = repeat([d.tgt.vocab.eos], y_pad_length)
        push!(padded_x, append!(x_pad_seq, sent_pair[1]))
        push!(padded_y, append!(sent_pair[2], y_pad_seq))
    end
    
    no_of_sentences = length(padded_x)

    
    padded_x = permutedims(hcat(padded_x...), (2,1))
    padded_y = permutedims(hcat(padded_y...), (2,1))
    
    return (padded_x,padded_y)
end

@info "Testing MTData"
dtrn = MTData(tr_train, en_train)
ddev = MTData(tr_dev, en_dev)
dtst = MTData(tr_test, en_test)
x,y = first(dtst)

@test length(collect(dtst)) == 48
@test size.((x,y)) == ((128,10),(128,24))
@test x[1,1] == tr_vocab.eos
@test x[1,end] != tr_vocab.eos
@test y[1,1] == en_vocab.eos;
@test y[1,2] != en_vocab.eos
@test y[1,end] == en_vocab.eos

mutable struct S2S_v1
    srcembed::Embed     # source language embedding
    encoder::RNN        # encoder RNN (can be bidirectional)
    tgtembed::Embed     # target language embedding
    decoder::RNN        # decoder RNN
    projection::Linear  # converts decoder output to vocab scores
    dropout::Real       # dropout probability to prevent overfitting
    srcvocab::Vocab     # source language vocabulary
    tgtvocab::Vocab     # target language vocabulary
end

function S2S_v1(hidden::Int,         # hidden size for both the encoder and decoder RNN
                srcembsz::Int,       # embedding size for source language
                tgtembsz::Int,       # embedding size for target language
                srcvocab::Vocab,     # vocabulary for source language
                tgtvocab::Vocab;     # vocabulary for target language
                layers=1,            # number of layers
                bidirectional=false, # whether encoder RNN is bidirectional
                dropout=0)           # dropout probability
    
    srcembed = Embed(length(srcvocab.i2w), srcembsz)
    tgtembed = Embed(length(tgtvocab.i2w), tgtembsz)
    decoder_layers = layers
    if(bidirectional == true)
        decoder_layers = 2 * layers
    end
    
    encoder = RNN(srcembsz, hidden, rnnType = :lstm, bidirectional = bidirectional, dropout = dropout, numLayers = layers, h = 0)
    decoder = RNN(tgtembsz, hidden, rnnType = :lstm, dropout = dropout, numLayers = decoder_layers, h = 0)
    projection = Linear(hidden, length(tgtvocab.i2w))
    
    S2S_v1(srcembed, encoder, tgtembed, decoder, projection, dropout, srcvocab, tgtvocab)
    
end 

function (s::S2S_v1)(src, tgt; average=true)
    src_embed_tensor = s.srcembed(src)
    s.encoder.h = 0
    s.encoder.c = 0
    y_enc = s.encoder(src_embed_tensor)
    tgt_embed_tensor = s.tgtembed(tgt[:,1:end-1])
    s.decoder.h = copy(s.encoder.h)
    s.decoder.c = copy(s.encoder.c)
    y_dec = s.decoder(tgt_embed_tensor)
    hy, b ,ty = size(y_dec)
    y_dec = reshape(y_dec, (hy, b*ty))
    scores = s.projection(y_dec)
    #check dropout
    y_gold = mask!(tgt[:,2:end], s.tgtvocab.eos)
    nll(scores, y_gold; average = average)
end

@info "Testing S2S_v1"
Knet.seed!(1)

model = S2S_v1(512, 512, 512, tr_vocab, en_vocab; layers=2, bidirectional=true, dropout=0.2)
(x,y) = first(dtst)
# Your loss can be slightly different due to different ordering of words in the vocabulary.
# The reference vocabulary starts with eos, unk, followed by words in decreasing frequency.
@test model(x,y; average=false)[2] == (14097.471f0, 1432)[2] #our version
@test model(x,y; average=false)[1] ≈ (14097.471f0, 1432)[1]
#@test model(x,y; average=false) == (14097.471f0, 1432) ,original

function loss(model, data; average=true)
    instances = 0
    cumulative_loss = 0
    for batch in data
        x, y = batch
        batch_loss, batch_instances = model(x,y; average=false)
        cumulative_loss += batch_loss
        instances += batch_instances
    end
    if (average)
        cumulative_loss / instances
    else
        cumulative_loss, instances
    end
end

@info "Testing loss"
#@test loss(model, dtst, average=false) == (1.0429117f6, 105937) ,true
@test loss(model, dtst, average=false)[1] ≈ (1.0429117f6, 105937)[1] #our version
@test loss(model, dtst, average=false)[2] == (1.0429117f6, 105937)[2] #our version
# Your loss can be slightly different due to different ordering of words in the vocabulary.
# The reference vocabulary starts with eos, unk, followed by words in decreasing frequency.
# Also, because we do not mask src, different batch sizes may lead to slightly different
# losses. The test above gives (1.0429178f6, 105937) with batchsize==1.

function train!(model, trn, dev, tst...)

    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), steps=100) do y
        losses = [ loss(model, d) for d in (dev,tst...) ]
        if losses[1] < bestloss
            bestmodel, bestloss = deepcopy(model), losses[1]
        end
        return (losses...,)
    end
    return bestmodel
end

@info "Training S2S_v1"

model = Knet.load("s2s_v1.jld2","model")
epochs = 1
ctrn = collect(dtrn)
trnx10 = collect(flatten(shuffle!(ctrn) for i in 1:epochs))
trn20 = ctrn[1:20]
dev38 = collect(ddev)
# Uncomment this to train the model (This takes about 30 mins on a V100):
#model = train!(model, trnx10, dev38, trn20)
# Uncomment this to save the model:
#Knet.save("s2s_v2.jld2","model",model)
# Uncomment this to load the model:
#model = Knet.load("s2s_vDY.jld2","model")

function (s::S2S_v1)(src::Matrix{Int}; stopfactor = 3)

    isDone = false
    batch_size = size(src,1)
    first_input = repeat([s.tgtvocab.eos], batch_size)
    is_all_finished = zeros(batch_size)
    translated_sentences = copy(first_input)
    max_length_output = 0
    s.encoder.h = 0
    s.encoder.c = 0
    src_embed_tensor = s.srcembed(src)
    y_enc = s.encoder(src_embed_tensor)
    s.decoder.h = copy(s.encoder.h)
    s.decoder.c = copy(s.encoder.c)
    input = first_input
    
    while (!isDone && max_length_output < stopfactor*size(src,2))
        
        
        tgt_embed_tensor = s.tgtembed(input)
        y = s.decoder(tgt_embed_tensor)
    
        scores = s.projection(y)
        
        
        output_words = reshape(map(x->x[1], argmax(scores, dims = 1)), batch_size)
        translated_sentences = hcat(translated_sentences, output_words')
        max_length_output = size(translated_sentences, 2)
        input = output_words

        
        tmp_output_words = copy(output_words)
        tmp_output_words = tmp_output_words .== s.tgtvocab.eos
        is_all_finished += tmp_output_words
        if(sum(is_all_finished.==0)==0)
            isDone = true
        end
    end
    
    return translated_sentences
end

# Utility to convert int arrays to sentence strings
function int2str(y,vocab)
    y = vec(y)
    ysos = findnext(w->!isequal(w,vocab.eos), y, 1)
    ysos == nothing && return ""
    yeos = something(findnext(isequal(vocab.eos), y, ysos), 1+length(y))
    join(vocab.i2w[y[ysos:yeos-1]], " ")
end

@info "Generating some translations"
d = MTData(tr_dev, en_dev, batchsize=1) |> collect
(src,tgt) = rand(d)
out = model(src)
println("SRC: ", int2str(src,model.srcvocab))
println("REF: ", int2str(tgt,model.tgtvocab))
println("OUT: ", int2str(out,model.tgtvocab))
# Here is a sample output:
# SRC: çin'e 15 şubat 2006'da ulaştım .
# REF: i made it to china on february 15 , 2006 .
# OUT: i got to china , china , at the last 15 years .

function bleu(s2s,d::MTData)
    d = MTData(d.src,d.tgt,batchsize=1)
    reffile = d.tgt.file
    hypfile,hyp = mktemp()
    for (x,y) in progress(collect(d))
        g = s2s(x)
        for i in 1:size(y,1)
            println(hyp, int2str(g[i,:], d.tgt.vocab))
        end
    end
    close(hyp)
    isfile("multi-bleu.perl") || download("https://github.com/moses-smt/mosesdecoder/raw/master/scripts/generic/multi-bleu.perl", "multi-bleu.perl")
    run(pipeline(`cat $hypfile`,`perl multi-bleu.perl $reffile`))
    return hypfile
end

@info "Calculating BLEU"
bleu(model, ddev)
