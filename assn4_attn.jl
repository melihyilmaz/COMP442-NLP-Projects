
import Pkg
using Pkg
Pkg.add("Knet"); Pkg.add("CuArrays"), Pkg.add("Random")

using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, IterTools
using CuArrays

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
    l.w * x .+ l.b #?
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
function MTData(src::TextReader, tgt::TextReader; batchmaker = arraybatch, batchsize = 64, maxlength = typemax(Int),
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

function int2str(y,vocab)
    y = vec(y)
    ysos = findnext(w->!isequal(w,vocab.eos), y, 1)
    ysos == nothing && return ""
    yeos = something(findnext(isequal(vocab.eos), y, ysos), 1+length(y))
    join(vocab.i2w[y[ysos:yeos-1]], " ")
end


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

struct Memory; w; end

struct Attention; wquery; wattn; scale; end

struct S2S
    srcembed::Embed       # encinput(B,Tx) -> srcembed(Ex,B,Tx)
    encoder::RNN          # srcembed(Ex,B,Tx) -> enccell(Dx*H,B,Tx)
    memory::Memory        # enccell(Dx*H,B,Tx) -> keys(H,Tx,B), vals(Dx*H,Tx,B)
    tgtembed::Embed       # decinput(B,Ty) -> tgtembed(Ey,B,Ty)
    decoder::RNN          # tgtembed(Ey,B,Ty) . attnvec(H,B,Ty)[t-1] = (Ey+H,B,Ty) -> deccell(H,B,Ty)
    attention::Attention  # deccell(H,B,Ty), keys(H,Tx,B), vals(Dx*H,Tx,B) -> attnvec(H,B,Ty)
    projection::Linear    # attnvec(H,B,Ty) -> proj(Vy,B,Ty)
    dropout::Real         # dropout probability
    srcvocab::Vocab       # source language vocabulary
    tgtvocab::Vocab       # target language vocabulary
end

if !isdefined(Main, :pretrained) || pretrained === nothing
    @info "Loading reference model"
    isfile("s2smodel.jld2") || download("http://people.csail.mit.edu/deniz/comp542/s2smodel.jld2","s2smodel.jld2")
    pretrained = Knet.load("s2smodel.jld2","model")
end
datadir = "datasets/tr_to_en"
if !isdir(datadir)
    @info "Downloading data"
    download("http://www.phontron.com/data/qi18naacl-dataset.tar.gz", "qi18naacl-dataset.tar.gz")
    run(`tar xzf qi18naacl-dataset.tar.gz`)
end
if !isdefined(Main, :tr_vocab)
    BATCHSIZE, MAXLENGTH = 64, 50
    @info "Reading data"
    tr_vocab = pretrained.srcvocab # Vocab("$datadir/tr.train", mincount=5)
    en_vocab = pretrained.tgtvocab # Vocab("$datadir/en.train", mincount=5)
    tr_train = TextReader("$datadir/tr.train", tr_vocab)
    en_train = TextReader("$datadir/en.train", en_vocab)
    tr_dev = TextReader("$datadir/tr.dev", tr_vocab)
    en_dev = TextReader("$datadir/en.dev", en_vocab)
    tr_test = TextReader("$datadir/tr.test", tr_vocab)
    en_test = TextReader("$datadir/en.test", en_vocab)
    dtrn = MTData(tr_train, en_train, batchsize=BATCHSIZE, maxlength=MAXLENGTH)
    ddev = MTData(tr_dev, en_dev, batchsize=BATCHSIZE)
    dtst = MTData(tr_test, en_test, batchsize=BATCHSIZE)
end

function S2S(hidden::Int, srcembsz::Int, tgtembsz::Int, srcvocab::Vocab, tgtvocab::Vocab;
             layers=1, bidirectional=false, dropout=0)
    
    srcembed = Embed(length(srcvocab.i2w), srcembsz)
    tgtembed = Embed(length(tgtvocab.i2w), tgtembsz)
    decoder_layers = layers
    memory_w = 1
    attn_wq = 1
    attn_scale = param(1)
    wattn = Linear(hidden, 2*hidden)
    if(bidirectional == true)
        encoder_layers = layers/2
        memory_w = param(hidden, 2*hidden)
        wattn = param(hidden, 3*hidden)
    end
    memory = Memory(memory_w)
    attention = Attention(attn_wq, wattn, attn_scale)
    
    
    encoder = RNN(srcembsz, hidden, rnnType = :lstm, bidirectional = bidirectional, dropout = dropout, numLayers = encoder_layers, h = 0)
    decoder = RNN(tgtembsz+hidden, hidden, rnnType = :lstm, dropout = dropout, numLayers = layers, h = 0)
    projection = Linear(hidden, length(tgtvocab.i2w))
    
    S2S(srcembed, encoder, memory, tgtembed, decoder, attention, projection, dropout, srcvocab, tgtvocab)
end

@testset "Testing S2S constructor" begin
    H,Ex,Ey,Vx,Vy,L,Dx,Pdrop = 8,9,10,length(dtrn.src.vocab.i2w),length(dtrn.tgt.vocab.i2w),2,2,0.2
    m = S2S(H,Ex,Ey,dtrn.src.vocab,dtrn.tgt.vocab;layers=L,bidirectional=(Dx==2),dropout=Pdrop)
    @test size(m.srcembed.w) == (Ex,Vx)
    @test size(m.tgtembed.w) == (Ey,Vy)
    @test m.encoder.inputSize == Ex
    @test m.decoder.inputSize == Ey + H
    @test m.encoder.hiddenSize == m.decoder.hiddenSize == H
    @test m.encoder.direction == Dx-1
    @test m.encoder.numLayers == (Dx == 2 ? L÷2 : L)
    @test m.decoder.numLayers == L
    @test m.encoder.dropout == m.decoder.dropout == Pdrop
    @test size(m.projection.w) == (Vy,H)
    @test size(m.memory.w) == (Dx == 2 ? (H,2H) : ())
    @test m.attention.wquery == 1
    @test size(m.attention.wattn) == (Dx == 2 ? (H,3H) : (H,2H))
    @test size(m.attention.scale) == (1,)
    @test m.srcvocab === dtrn.src.vocab
    @test m.tgtvocab === dtrn.tgt.vocab
end

function (m::Memory)(x)
    vals = permutedims(x, (1,3,2))
    keys = mmul(m.w, vals)
    return keys, vals
end

mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))

@testset "Testing memory" begin
    H,D,B,Tx = pretrained.encoder.hiddenSize, pretrained.encoder.direction+1, 4, 5
    x = KnetArray(randn(Float32,H*D,B,Tx))
    k,v = pretrained.memory(x)
    @test v == permutedims(x,(1,3,2))
    @test k == mmul(pretrained.memory.w, v)
end

function encode(s::S2S, src)
    src_embed_tensor = dropout(s.srcembed(src), s.dropout)
    s.encoder.h = 0
    s.encoder.c = 0
    y_enc = s.encoder(src_embed_tensor)
    s.decoder.h = s.encoder.h
    s.decoder.c = s.encoder.c
    
    keys, values = s.memory(y_enc)
    return keys, values
end

@testset "Testing encoder" begin
    src1,tgt1 = first(dtrn)
    key1,val1 = encode(pretrained, src1)
    H,D,B,Tx = pretrained.encoder.hiddenSize, pretrained.encoder.direction+1, size(src1,1), size(src1,2)
    @test size(key1) == (H,Tx,B)
    @test size(val1) == (H*D,Tx,B)
    @test (pretrained.decoder.h,pretrained.decoder.c) === (pretrained.encoder.h,pretrained.encoder.c)
    @test norm(key1) ≈ 1214.4755f0
    @test norm(val1) ≈ 191.10411f0
    @test norm(pretrained.decoder.h) ≈ 48.536964f0
    @test norm(pretrained.decoder.c) ≈ 391.69028f0
end

function (a::Attention)(cell, mem)
    keys, values = mem
    query = permutedims(mmul(a.wquery, cell), (3,1,2))
    scores = bmm(query, keys)
    scores = mmul(a.scale[1], scores)
    
    scores = softmax(scores, dims = 2)
    context = bmm(values, permutedims(scores, (2,1,3)))
    mmul(a.wattn, vcat(cell,permutedims(context, (1,3,2))))
end

@testset "Testing attention" begin
    src1,tgt1 = first(dtrn)
    key1,val1 = encode(pretrained, src1)
    H,B = pretrained.encoder.hiddenSize, size(src1,1)
    Knet.seed!(1)
    x = KnetArray(randn(Float32,H,B,5))
    y = pretrained.attention(x, (key1, val1))
    @test size(y) == size(x)
    @test norm(y) ≈ 808.381f0
end

function decode(s::S2S, tgt, mem, prev)
    
    tgt_embed_tensor = dropout(s.tgtembed(tgt), s.dropout)
    input = vcat(tgt_embed_tensor,prev)
    y_dec = s.decoder(input)
    s.attention(y_dec, mem)
end

@testset "Testing decoder" begin
    src1,tgt1 = first(dtrn)
    key1,val1 = encode(pretrained, src1)
    H,B = pretrained.encoder.hiddenSize, size(src1,1)
    Knet.seed!(1)
    cell = randn!(similar(key1, size(key1,1), size(key1,3), 1))
    cell = decode(pretrained, tgt1[:,1:1], (key1,val1), cell)
    @test size(cell) == (H,B,1)
    @test norm(cell) ≈ 131.21631f0
end

function (s::S2S)(src, tgt; average=true)
    batchsize = size(tgt,1)
    
    mem = encode(s, src)
    
    prev = zeros(Float32, size(s.projection.w, 2), batchsize, 1)
    
    if(gpu()>=0)
        prev = KnetArray(prev)
    end
    
    output = copy(prev)
    
    for i = 1:size(tgt,2)-1
        tmp_tgt = reshape(tgt[:,i], (size(tgt[:,i], 1), 1))
        y_dec = decode(s, tmp_tgt, mem, prev)
        prev = y_dec
        output = cat(output, y_dec, dims = 3)
    end

    output = output[:,:,2:end]
    hy, b ,ty = size(output)
    
    output = reshape(output, (hy, b*ty))
    
    scores = s.projection(output)
    y_gold = mask!(tgt[:,2:end], s.tgtvocab.eos)
    
    nll(scores, y_gold; average = average)
end

@testset "Testing loss" begin
    src1,tgt1 = first(dtrn)
    @test pretrained(src1,tgt1) ≈ 1.4666592f0
    @test pretrained(src1,tgt1,average=false)[2] == (1949.1901f0, 1329)[2]
    @test pretrained(src1,tgt1,average=false)[1] ≈ (1949.1901f0, 1329)[1] #converted loss to similarity
end

function (s::S2S)(src; stopfactor = 3)
    
    
    isDone = false
    batch_size = size(src,1)
    input = repeat([s.tgtvocab.eos], batch_size)
    is_all_finished = zeros(batch_size)
    translated_sentences = copy(input)
    max_length_output = 0
    
    mem = encode(s, src)
    
    prev_decoder_output = zeros(Float32, size(s.encoder.h, 1), batch_size, 1)
    if (gpu() >= 0)
        prev_decoder_output = KnetArray(prev_decoder_output)
    end
    input = reshape(input, (length(input), 1))
    
    while (!isDone && max_length_output < stopfactor*size(src,2))        
        
        
        y = decode(s, input, mem, prev_decoder_output)
        prev_decoder_output = y
        
          
        hy, b ,ty = size(y)
        y = reshape(y, (hy, b*ty))
        
        scores = s.projection(y)
        
        output_words = reshape(map(x->x[1], argmax(scores, dims = 1)), batch_size)
        translated_sentences = hcat(translated_sentences, output_words)
       
        max_length_output = size(translated_sentences, 2)
        input = reshape(output_words, (length(output_words), 1))
        
       
        tmp_output_words = copy(output_words)
        tmp_output_words = tmp_output_words .== s.tgtvocab.eos
        is_all_finished += tmp_output_words
        if(sum(is_all_finished.==0)==0)
            isDone = true
        end
    end
    return translated_sentences[:, 2:end]
end

@testset "Testing translator" begin
    src1,tgt1 = first(dtrn)
    tgt2 = pretrained(src1)
    @test size(tgt2) == (64, 41)
    @test tgt2[1:3,1:3] == [14 25 10647; 37 25 1426; 27 5 349]
end

function trainmodel(trn,                  # Training data
                    dev,                  # Validation data, used to determine the best model
                    tst...;               # Zero or more test datasets, their loss will be periodically reported
                    bidirectional = true, # Whether to use a bidirectional encoder
                    layers = 2,           # Number of layers (use `layers÷2` for a bidirectional encoder)
                    hidden = 512,         # Size of the hidden vectors
                    srcembed = 512,       # Size of the source language embedding vectors
                    tgtembed = 512,       # Size of the target language embedding vectors
                    dropout = 0.2,        # Dropout probability
                    epochs = 0,           # Number of epochs (one of epochs or iters should be nonzero for training)
                    iters = 0,            # Number of iterations (one of epochs or iters should be nonzero for training)
                    bleu = false,         # Whether to calculate the BLEU score for the final model
                    save = false,         # Whether to save the final model
                    seconds = 60,         # Frequency of progress reporting
                    )
    @show bidirectional, layers, hidden, srcembed, tgtembed, dropout, epochs, iters, bleu, save; flush(stdout)
    model = S2S(hidden, srcembed, tgtembed, trn.src.vocab, trn.tgt.vocab;
                layers=layers, dropout=dropout, bidirectional=bidirectional)
    
    epochs == iters == 0 && return model

    (ctrn,cdev,ctst) = collect(trn),collect(dev),collect.(tst)
    traindata = (epochs > 0
                 ? collect(flatten(shuffle!(ctrn) for i in 1:epochs))
                 : shuffle!(collect(take(cycle(ctrn), iters))))

    bestloss, bestmodel = loss(model, cdev), deepcopy(model)
    progress!(adam(model, traindata), seconds=seconds) do y
        devloss = loss(model, cdev)
        tstloss = map(d->loss(model,d), ctst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (dev=devloss, tst=tstloss, mem=Float32(CuArrays.usage[]))
    end
    save && Knet.save("attn-$(Int(time_ns())).jld2", "model", bestmodel)
    bleu && Main.bleu(bestmodel,dev)
    return bestmodel
end

# Uncomment the appropriate option for training:

#model = pretrained  # Use reference model
#model = Knet.load("attn-2888149734332.jld2", "model")  # Load pretrained model
model = trainmodel(dtrn,ddev,take(dtrn,20); epochs=10, save=true, bleu=true)  # Train model

data1 = MTData(tr_dev, en_dev, batchsize=1) |> collect;
function translate_sample(model, data)
    (src,tgt) = rand(data)
    out = model(src)
    println("SRC: ", int2str(src,model.srcvocab))
    println("REF: ", int2str(tgt,model.tgtvocab))
    println("OUT: ", int2str(out,model.tgtvocab))
end

translate_sample(model, data1)

function translate_input(model)
    v = model.srcvocab
    src = [ get(v.w2i, w, v.unk) for w in v.tokenizer(readline()) ]'
    out = model(src)
    println("SRC: ", int2str(src,model.srcvocab))
    println("OUT: ", int2str(out,model.tgtvocab))
end

# translate_input(model)
