%ham_train contains the occurrences of each word in ham emails. 1-by-N vector
ham_train = csvread('ham_train.csv');
%spam_train contains the occurrences of each word in spam emails. 1-by-N vector
spam_train = csvread('spam_train.csv');
%N is the size of vocabulary.
N = size(ham_train, 2);
%There 9034 ham emails and 3372 spam emails in the training samples
num_ham_train = 9034;
num_spam_train = 3372;
%Do smoothing
x = [ham_train;spam_train] + 1;

%ham_test contains the occurences of each word in each ham test email. P-by-N vector, with P is number of ham test emails.
load ham_test.txt;
ham_test_tight = spconvert(ham_test);
ham_test = sparse(size(ham_test_tight, 1), size(ham_train, 2));
ham_test(:, 1:size(ham_test_tight, 2)) = ham_test_tight;
%spam_test contains the occurences of each word in each spam test email. Q-by-N vector, with Q is number of spam test emails.
load spam_test.txt;
spam_test_tight = spconvert(spam_test);
spam_test = sparse(size(spam_test_tight, 1), size(spam_train, 2));
spam_test(:, 1:size(spam_test_tight, 2)) = spam_test_tight;

%TODO
%Implement a ham/spam email classifier, and calculate the accuracy of your classifier
l = x ./ repmat(sum(x, 2) , [1, size(x,2)]);
[r, I] = sort(l(2, :) ./ l(1, :));
top = I(end-9: end);

prior = [num_ham_train/(num_spam_train + num_ham_train); num_spam_train/(num_spam_train + num_ham_train)];
logl = log(l);
fn = 0;
fp = 0;
P = size(ham_test, 1);
Q = size(spam_test, 1);
for i = 1:P
   post = sum(repmat(ham_test(i, :), [2, 1]) .* logl, 2) + log(prior);
   if post(1) < post(2)
       fn = fn +1;
   end
end
for i = 1:Q
   post = sum(repmat(spam_test(i, :), [2, 1]) .* logl, 2) + log(prior);
   if post(1) > post(2)
       fp = fp + 1;
   end
end
err = (fp + fn)/(P + Q);
precison = (Q -  fp)/Q;
recall = (Q -  fp)/(Q -  fp + fn);
fprintf(1, 'test error using naive Bayes model for text classification: %6.2f\n', err);
fprintf(1, 'precison using naive Bayes model for text classification: %6.2f\n', precison);
fprintf(1, 'recall using naive Bayes model for text classification: %6.2f\n', recall);
