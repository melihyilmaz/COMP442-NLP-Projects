
import Pkg; Pkg.add("Knet"); Pkg.add("IterTools"); Pkg.add("StatsBase")

using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test
macro size(z, s); esc(:(@assert (size($z) == $s) string(summary($z),!=,$s))); end # for debugging

const datadir = "nn4nlp-code/data/ptb"
isdir(datadir) || run(`git clone https://github.com/neubig/nn4nlp-code.git`)

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

@info "Testing Vocab"
f = "$datadir/train.txt"
v = Vocab(f)
@test all(v.w2i[w] == i for (i,w) in enumerate(v.i2w))
@test length(Vocab(f).i2w) == 10000
@test length(Vocab(f, vocabsize=1234).i2w) == 1234
@test length(Vocab(f, mincount=5).i2w) == 9859

train_vocab = Vocab("$datadir/train.txt")

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

@info "Testing TextReader"
train_sentences, valid_sentences, test_sentences =
    (TextReader("$datadir/$file.txt", train_vocab) for file in ("train","valid","test"))
@test length(first(train_sentences)) == 24
@test length(collect(train_sentences)) == 42068
@test length(collect(valid_sentences)) == 3370
@test length(collect(test_sentences)) == 3761

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

@info "Testing Embed"
Knet.seed!(1)
embed = Embed(100,10)
input = rand(1:100, 2, 3)
output = embed(input)
@test size(output) == (10, 2, 3)
@test norm(output) ≈ 0.59804f0

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    w = param(outputsize, inputsize)
    b = param0(outputsize)
    Linear(w,b)
end

function (l::Linear)(x)
    l.w * x .+ l.b
end

@info "Testing Linear"
Knet.seed!(1)
linear = Linear(100,10)
input = oftype(linear.w, randn(Float32, 100, 5))
output = linear(input)
@test size(output) == (10, 5)
@test norm(output) ≈ 5.5301356f0

struct NNLM; vocab; windowsize; embed; hidden; output; dropout; end

function NNLM(vocab::Vocab, windowsize::Int, embedsize::Int, hiddensize::Int, dropout::Real)
    vocabsize = length(vocab.i2w)
    embed = Embed(vocabsize, embedsize)
    hidden = Linear(windowsize * embedsize, hiddensize)
    output = Linear(hiddensize, vocabsize)    
    NNLM(vocab, windowsize, embed, hidden, output, dropout)
end

# Default model parameters
HIST = 3
EMBED = 128
HIDDEN = 128
DROPOUT = 0.5
VOCAB = length(train_vocab.i2w)


@info "Testing NNLM"
model = NNLM(train_vocab, HIST, EMBED, HIDDEN, DROPOUT)
@test model.vocab === train_vocab
@test model.windowsize === HIST
@test size(model.embed.w) == (EMBED,VOCAB)
@test size(model.hidden.w) == (HIDDEN,HIST*EMBED)
@test size(model.hidden.b) == (HIDDEN,)
@test size(model.output.w) == (VOCAB,HIDDEN)
@test size(model.output.b) == (VOCAB,)
@test model.dropout == 0.5


function pred_v1(m::NNLM, hist::AbstractVector{Int})
    @assert length(hist) == m.windowsize
    
    m.output(tanh.(dropout(m.hidden(dropout(vec(m.embed(hist)), m.dropout)), m.dropout)))
end

@info "Testing pred_v1"
h = repeat([model.vocab.eos], model.windowsize)
p = pred_v1(model, h)
@test size(p) == size(train_vocab.i2w)


# This predicts the scores for the whole sentence, will be used for later testing.
function scores_v1(model, sent)
    hist = repeat([ model.vocab.eos ], model.windowsize)
    scores = []
    for word in [ sent; model.vocab.eos ]
        push!(scores, pred_v1(model, hist))
        hist = [ hist[2:end]; word ]
    end
    hcat(scores...)
end

sent = first(train_sentences)
@test size(scores_v1(model, sent)) == (length(train_vocab.i2w), length(sent)+1)

function generate(m::NNLM; maxlength=30)
    sentence = []
    history = [m.vocab.eos]
    history = repeat(history, m.windowsize)
    scores = pred_v1(m, history)
    random = rand()
    probs = softmax(scores)
    cum_sum = 0
    pred_word = 0
    for i = 1:length(probs)
        cum_sum += probs[i]
        if random <= cum_sum
            pred_word = i
            break
        end
    end
    
    while (pred_word != m.vocab.eos && length(sentence) < maxlength)  
        push!(history, pred_word)
        push!(sentence, pred_word)
        deleteat!(history, 1)
        scores = pred_v1(m, history)
        random = rand()
        probs = softmax(scores)
        cum_sum = 0
        for i = 1:length(probs)
            cum_sum += probs[i]
            if random <= cum_sum
                pred_word = i
                break
            end
        end
    end
    
    # if(pred_word == m.vocab.eos) push!(sentence, pred_word) to display eos token
    
    sentence = m.vocab.i2w[sentence]
    sentence = join(sentence, " ")
    return sentence
end

@info "Testing generate"
s = generate(model, maxlength=5)
@test s isa String
@test length(split(s)) <= 5



function loss_v1(m::NNLM, sent::AbstractVector{Int}; average = true)
    # Your code here
    total_loss = 0
    history = [m.vocab.eos]; history = repeat(history, m.windowsize)
       
    for word in sent  
        prob = softmax(pred_v1(m, history))
        total_loss -= log(prob[word])
        push!(history, word)
        deleteat!(history, 1)
    end
    
    prob = softmax(pred_v1(m, history))
    total_loss -= log(prob[m.vocab.eos])
    
    
    if(average == true) 
        return total_loss / (length(sent) + 1)
    end
    return (total_loss, length(sent) + 1)
end

@info "Testing loss_v1"
s = first(train_sentences)
avgloss = loss_v1(model,s)
(tot, cnt) = loss_v1(model, s, average = false)
@test 9 < avgloss < 10
@test cnt == length(s) + 1
@test tot/cnt ≈ avgloss

function maploss(lossfn, model, data; average = true)
    word_count = 0
    total_loss = 0
    for sentence in data
        (tot, cnt) = lossfn(model, sentence, average = false)
        total_loss += tot
        word_count += cnt
    end
    if average == true 
        return total_loss/word_count
    else
        return (total_loss,word_count)
    end
    
end

@info "Testing maploss"
tst100 = collect(take(test_sentences, 100))
avgloss = maploss(loss_v1, model, tst100)
@test 9 < avgloss < 10
(tot, cnt) = maploss(loss_v1, model, tst100, average = false)
@test cnt == length(tst100) + sum(length.(tst100))
@test tot/cnt ≈ avgloss

@info "Timing loss_v1 with 1000 sentences"
tst1000 = collect(take(test_sentences, 1000))
@time maploss(loss_v1, model, tst1000)

@info "Timing loss_v1 training with 100 sentences"
trn100 = ((model,x) for x in collect(take(train_sentences, 100)))
@time sgd!(loss_v1, trn100)

function pred_v2(m::NNLM, hist::AbstractMatrix{Int})
    sentence_length = size(hist)[2]
    m.output(tanh.(dropout(m.hidden(dropout(reshape(m.embed(hist), (:, sentence_length)), m.dropout)), m.dropout)))
end

@info "Testing pred_v2"

function scores_v2(model, sent)
    hist = [ repeat([ model.vocab.eos ], model.windowsize); sent ]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
    @assert size(hist) == (model.windowsize, length(sent)+1)
    return pred_v2(model, hist)
end

sent = first(test_sentences)
s1, s2 = scores_v1(model, sent), scores_v2(model, sent)
@test size(s1) == size(s2) == (length(train_vocab.i2w), length(sent)+1)
@test s1 ≈ s2

function loss_v2(m::NNLM, sent::AbstractVector{Int}; average = true)
    
    correct_answers = []
    word_history = [m.vocab.eos]; word_history = repeat(word_history, m.windowsize)
    hist = copy(word_history)
       
    for word in sent  
        push!(correct_answers, word)
        push!(word_history, word)
        deleteat!(word_history, 1)
        hist  = hcat(hist, word_history)
    end
    
    push!(correct_answers, m.vocab.eos)
    correct_answers = convert(Array{Int,1}, correct_answers)
    return nll(pred_v2(m, hist), correct_answers; average = average)
end




@info "Testing loss_v2"
s = first(test_sentences)
@test loss_v1(model, s) ≈ loss_v2(model, s)
tst100 = collect(take(test_sentences, 100))
@test maploss(loss_v1, model, tst100) ≈ maploss(loss_v2, model, tst100)

@info "Timing loss_v2  with 10K sentences"
tst10k = collect(take(train_sentences, 10000))
@time maploss(loss_v2, model, tst10k)

@info "Timing loss_v2 training with 1000 sentences"
trn1k = ((model,x) for x in collect(take(train_sentences, 1000)))
@time sgd!(loss_v2, trn1k)

function pred_v3(m::NNLM, hist::Array{Int})
    
    window_length, batch_length, sentence_length = size(hist)
    embed_output = reshape(m.embed(hist), (:, batch_length*sentence_length))
    output = m.output(tanh.(dropout(m.hidden(dropout(embed_output, m.dropout)), m.dropout)))
    return reshape(output, (:, batch_length, sentence_length))
end

@info "Testing pred_v3"

function scores_v3(model, sent)
    hist = [ repeat([ model.vocab.eos ], model.windowsize); sent ]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
    @assert size(hist) == (model.windowsize, length(sent)+1)
    hist = reshape(hist, size(hist,1), 1, size(hist,2))
    return pred_v3(model, hist)
end

sent = first(train_sentences)
@test scores_v2(model, sent) ≈ scores_v3(model, sent)[:,1,:]

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

@info "Testing mask!"
a = [1 2 1 1 1; 2 2 2 1 1; 1 1 2 2 2; 1 1 2 2 1]
@test mask!(a,1) == [1 2 1 0 0; 2 2 2 1 0; 1 1 2 2 2; 1 1 2 2 1]

function loss_v3(m::NNLM, batch::AbstractMatrix{Int}; average = true)

    correct_answers = []
    
    batch_length, sentence_length = size(batch)
    hist = []
   
    for sent in 1:batch_length
        word_history = [m.vocab.eos]; word_history = repeat(word_history, m.windowsize)
        
        for word in batch[sent, :]  
            append!(hist, word_history)
            append!(correct_answers, word)
            push!(word_history, word)
            deleteat!(word_history, 1)
            
        end
    end
        
    correct_answers = convert(Array{Int,1}, correct_answers)
    hist = convert(Array{Int,1}, hist)
    hist = permutedims(reshape(hist,(m.windowsize, sentence_length,batch_length)), [1 3 2])
    correct_answers = mask!(permutedims(reshape(correct_answers,(sentence_length,batch_length))), m.vocab.eos)
    return nll(pred_v3(m, hist), correct_answers; average = average)
    
end

@info "Testing loss_v3"
s = first(test_sentences)
b = [ s; model.vocab.eos ]'
@test loss_v2(model, s) ≈ loss_v3(model, b)

struct LMData
    src::TextReader
    batchsize::Int
    maxlength::Int
    bucketwidth::Int
    buckets
end

function LMData(src::TextReader; batchsize = 64, maxlength = typemax(Int), bucketwidth = 10)
    numbuckets = min(128, maxlength ÷ bucketwidth)
    buckets = [ [] for i in 1:numbuckets ]
    LMData(src, batchsize, maxlength, bucketwidth, buckets)
end

Base.IteratorSize(::Type{LMData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{LMData}) = Base.HasEltype()
Base.eltype(::Type{LMData}) = Matrix{Int}

function Base.iterate(d::LMData, state=nothing)
    if state == nothing
        for b in d.buckets; empty!(b); end
    end
    bucket,ibucket = nothing,nothing
    while true
        iter = (state === nothing ? iterate(d.src) : iterate(d.src, state))
        if iter === nothing
            ibucket = findfirst(x -> !isempty(x), d.buckets)
            bucket = (ibucket === nothing ? nothing : d.buckets[ibucket])
            break
        else
            sent, state = iter
            if length(sent) > d.maxlength || length(sent) == 0; continue; end
            ibucket = min(1 + (length(sent)-1) ÷ d.bucketwidth, length(d.buckets))
            bucket = d.buckets[ibucket]
            push!(bucket, sent)
            if length(bucket) === d.batchsize; break; end
        end
    end
    if bucket === nothing; return nothing; end
    batchsize = length(bucket)
    maxlen = maximum(length.(bucket))
    batch = fill(d.src.vocab.eos, batchsize, maxlen + 1)
    for i in 1:batchsize
        batch[i, 1:length(bucket[i])] = bucket[i]
    end
    empty!(bucket)
    return batch, state
end

@info "Timing loss_v2 and loss_v3 at various batch sizes"
@info loss_v2; test_collect = collect(test_sentences)
GC.gc(); @time p2 = maploss(loss_v2, model, test_collect)
for B in (1, 8, 16, 32, 64, 128, 256)
    @info loss_v3,B; test_batches = collect(LMData(test_sentences, batchsize = B))
    GC.gc(); @time p3 = maploss(loss_v3, model, test_batches); @test p3 ≈ p2
end

@info "Timing SGD for loss_v2 and loss_v3 at various batch sizes"
train(loss, model, data) = sgd!(loss, ((model,sent) for sent in data))
@info loss_v2; test_collect = collect(test_sentences)
GC.gc(); @time train(loss_v2, model, test_collect)
for B in (1, 8, 16, 32, 64, 128, 256)
    @info loss_v3,B; test_batches = collect(LMData(test_sentences, batchsize = B))
    GC.gc(); @time train(loss_v3, model, test_batches)
end

model = NNLM(train_vocab, HIST, EMBED, HIDDEN, DROPOUT)
train_batches = collect(LMData(train_sentences))
valid_batches = collect(LMData(valid_sentences))
test_batches = collect(LMData(test_sentences))
train_batches50 = train_batches[1:50] # Small sample for quick loss calculation

epoch = adam(loss_v3, ((model, batch) for batch in train_batches))
bestmodel, bestloss = deepcopy(model), maploss(loss_v3, model, valid_batches)

progress!(ncycle(epoch, 100), seconds=5) do x
    global bestmodel, bestloss
    # Report gradient norm for the first batch
    f = @diff loss_v3(model, train_batches[1])
    gnorm = sqrt(sum(norm(grad(f,x))^2 for x in params(model)))
    # Report training and validation loss
    trnloss = maploss(loss_v3, model, train_batches50)
    devloss = maploss(loss_v3, model, valid_batches)
    # Save model that does best on validation data
    if devloss < bestloss
        bestmodel, bestloss = deepcopy(model), devloss
    end
    (trn=exp(trnloss), dev=exp(devloss), ∇=gnorm)
end

# julia> generate(bestmodel)
# "the nasdaq composite index finished at N compared with ual earlier in the statement"
#
# julia> generate(bestmodel)
# "in the pentagon joseph r. waertsilae transactions the 1\\/2-year transaction was oversubscribed an analyst at <unk>"
