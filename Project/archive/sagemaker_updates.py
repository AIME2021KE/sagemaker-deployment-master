# TODO: Use this space to determine the five most frequently appearing words in the training set.
# FROM https://stackoverflow.com/questions/7971618/return-first-n-keyvalue-pairs-from-dict
print(list(word_dict)[:5])

# Use this cell to examine one of the processed reviews to make sure everything is working as intended.
print(train_X[100])


print(test_X[100])

**Question:** In the cells above we use the `preprocess_data` and `convert_and_pad_data` methods to process both the training and testing set. Why or why not might this be a problem?

**Answer:**
### ANSWER: the training and testing sets can have different vocabulary sets, but hopefully with our noword and infrequent word we can push those all away.

def train(model, train_loader, epochs, optimizer, loss_fn, device):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            outmodel = model(batch_X)
            loss = loss_fn(outmodel, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
        
# TODO: Deploy the trained model
estimator_predict = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

predictor = estimator_predict

**Question:** How does this model compare to the XGBoost model you created earlier? Why might these two models perform differently on this dataset? Which do *you* think is better for sentiment analysis?

**Answer:**
### ANSWER: The numbers are quite comparable, with XGBoost (Sentiment Analysis) where we got numbers like 0.87+ versus the 0.85- here. I would still think that XGBoost is probably better suited for this kind of problem as it seems designed for handling exactly this kind of problem (text analysis of sparsely filled matrix) , but the RNN/LSTM approach could take advantage of the correlation of positive words within the text.



# TODO: Convert test_review into a form usable by the model and save the results in test_data
test_data = None
test_data_raw = review_to_words(test_review)
test_data, test_data_len = convert_and_pad_data(word_dict, [test_data_raw])

#NOTE 12/27/2021: a number of print statements to better understand our output data structure with just a single review
#  we note that we needed to wrap the raw output of the review to words in another list: [test_data_raw] to get things to work 
#  right
print(len(test_review))
print(test_data_raw)
print(type(test_data))
print(test_data,len(test_data[0]))

predictor.predict(test_data)

