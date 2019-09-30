#NOTE: The entire scripte takes about 6-7 min. to run on the first try. Takes less than a min. in subsequent runs

# Get a list of review files to iterate over the filenames later
trn_pos = readdir(pwd()*"\\train\\pos")
trn_neg = readdir(pwd()*"\\train\\neg")
test_pos = readdir(pwd()*"\\test\\pos")
test_neg = readdir(pwd()*"\\test\\neg")

# Define a struct to serve as the 'value' in vocab dictionary. count field to store an array
#of word freq in pos and neg reviews. Int. id currently unused.
struct id_count
    id
    count
end

# To process raw review texts. All reviews are cropped/padded to fixed size
function str_process_pad(review,fix_size)
    review = (lowercase(review))
    review = replace(review,r"""[,.:;?!()]""" => "")
    review = replace(review,'\"' => "")
    review = replace(review,"<br /><br />" => "")
    review = split(review)
    curr_size = length(review)
    if curr_size >= fix_size
        return review[1:fix_size]
    else
        padding = fill("<pad>",(fix_size - curr_size))
        review =[review; padding]
        return review
    end
end

wdict = Dict()
w2idcount(x) = get!(wdict, x, id_count(1+length(wdict),[0 0]))
#Dummy placeholders for unknown words and end-of-sentece padding
UNK = w2idcount("<unk>")
PAD = w2idcount("<pad>")

#Constants to determine review size and the threshold for word appearences
fix_size = 300
min_num_appearences = 5

POS = 1
NEG =2

# Read, process training reviews and count words for further training
for file_name in trn_pos
    review = open(pwd()*"\\train\\pos\\"*file_name) do file
    read(file, String)
end
    review = str_process_pad(review,fix_size)
    w2idcount.(review)
    for word in review
        wdict[word].count[POS]+=1
    end
end

for file_name in trn_neg
    review = open(pwd()*"\\train\\neg\\"*file_name) do file
    read(file, String)
end
    review = str_process_pad(review,fix_size)
    w2idcount.(review)
    for word in review
        wdict[word].count[NEG]+=1
    end
end

#Deleting the words under an 'appearence threshold' from the vocab, adding their count to UNK
for (key,value) in wdict
    if sum(value.count)<min_num_appearences
        wdict["<unk>"].count[POS]+=wdict[key].count[1]
        wdict["<unk>"].count[NEG]+=wdict[key].count[2]
        delete!(wdict,key)
    end
    
    #Function to classify each review as pos/neg. Sum of log probabilities are used to avoid
#potential problems that could arise from the multip. of small numbers. Smoothing (+1 for)
#each word's count is also implemented

function pred_review(review,word_freq)
#First add prior probs
    logprob_pos = log(length(trn_pos)/(length(trn_pos)+length(trn_neg)))
    logprob_neg = log(length(trn_neg)/(length(trn_pos)+length(trn_neg)))
    
    num_words_pos = fix_size*length(trn_pos)
    num_words_neg = fix_size*length(trn_neg)
    
   for word in review
        counts = get(word_freq, word,word_freq["<unk>"]).count
        logprob_pos+= log((counts[POS]+1)/num_words_pos)
        logprob_neg+= log((counts[NEG]+1)/num_words_neg)
    end
    
    if logprob_pos>=logprob_neg
        return "Positive"
    else
        return "Negative"
    end
end

#Reading, processing, predicting and evaluating the predictions for test set
#3 min
TP=0
FN=0
TN=0
FP=0

#Positive test samples
for file_name in test_pos
    review = open(pwd()*"\\test\\pos\\"*file_name) do file
    read(file, String)
end
    review = str_process_pad(review,fix_size)
    pred_label = pred_review(review,wdict)
    if pred_label == "Positive"
        TP+=1
    else
        FN+=1
    end
end

#Negative test samples
for file_name in test_neg
    review = open(pwd()*"\\test\\neg\\"*file_name) do file
    read(file, String)
end
    review = str_process_pad(review,fix_size)
    pred_label = pred_review(review,wdict)
    if pred_label == "Positive"
        FP+=1
    else
        TN+=1
    end
end
end

acc = (TP+TN)/(TP+TN+FP+FN)
rec = TP/(TP+FN)
prec = TP/(TP+FP)
println("Test accuracy: " ,acc)
println("Test precision: ", prec)
print("Test recall: ",rec)

