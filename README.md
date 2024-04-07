# An exploration of the impact of Vectorization on Classification

## 1. Introduction
The surge of online reviews has become a powerful force in shaping consumer decisions, necessitating the use of automated sentiment classification to manage and interpret this abundant data effectively (Nguyen et al., 2018). This research examines how different text vectorization techniques affect the accuracy of sentiment classification models, specifically in the context of Amazon Customer Review Dataset. This is crucial for businesses aiming to harness customer insights for product development and market positioning (Tumu et al., 2020). With reviews having a proven impact on sales, the ability to efficiently analyze sentiment through Natural Language Processing stands as a vital tool for adapting to consumer needs (Kozhevnikov & Pankratova, 2020). This inquiry is pivotal in optimizing the process of gleaning meaningful information from extensive customer feedback.

## 2. Literature Review
In evaluating the effectiveness of different vectorization methods and machine learning techniques, the literature offers key insights. One study focused on classifying student-lecture comments via sentiment analysis and compared the performance of TF-IDF and Count Vectorization. It was observed that TF-IDF consistently outperformed Count Vectorization across various classifiers. Among the classifiers tested, which included Random Forest (RF), Support Vector Machine (SVM), Naïve-Bayes, Gradient Boosting, and Artificial Neural Networks (ANN), RF and ANN achieved the highest prediction accuracy at 97%. However, RF was noted for its computational efficiency and simplicity over the more complex ANN (Rakhmanov, 2020).

In a different analysis centered on drug review sentiment classification, the combination of N-gram and TF-IDF vectorization methods with machine learning models demonstrated significant success. The study achieved a high accuracy rate of 89% in sentiment analysis, with the Random Forest classifier delivering superior results compared to its counterparts. The use of N-grams notably enhanced the model's precision in detecting and classifying nuanced sentiments within drug reviews, highlighting the strength of merging these specific techniques for detailed sentiment analysis (Tumu et al., 2020).

In a study comparing Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) for analyzing Twitter data on COVID-19, it was found that LDA identified 14 broad topics that lacked specific relevance, whereas NMF identified 10 distinct and meaningful themes, including government responses and travel restrictions. LDA's wider topic range, despite encompassing geographical specifics, was overshadowed by its inclusion of more generic topics. Conversely, NMF's focused and clear topics underscore its suitability for capturing and understanding public sentiment more effectively (Egger & Yu, 2022).

In a comparative study on word embeddings for text classification, it was found that BERT outperforms other embeddings like ELMo, GloVe, FastText, and Word2Vec, especially in handling longer texts and enhancing accuracy in shorter datasets. However, BERT's superior performance comes with higher computational demands. ELMo offers benefits for short texts by capturing context-dependent meanings but doesn't consistently surpass classic embeddings in longer text scenarios, presenting computational challenges of its own. GloVe and FastText are highlighted for their effectiveness, with GloVe slightly outdoing Word2Vec in some cases. This underscores the importance of matching word embeddings to the task's computational constraints, with GloVe and FastText as efficient choices for limited resources, and BERT and ELMo for settings where computational capacity allows for deeper text analysis (Wang, Nulty, & Lillis, 2020).

## 3. Methodology

### 3.1 Data Exploration
Figure 3.1 provides a structured summary of the dataset used in the analysis, detailing the columns and their respective data types. The dataset comprises 294,240 entries across eight attributes, including overall rating, verification status, review text, and timestamps, among others. Notably, the 'summary' field has fewer non-null values than other columns, suggesting some entries may lack a review summary.

![Figure 3.1: Dataset Overview of Amazon Customer Reviews](/images/Figure1.png)

Figure 3.2 depicts the distribution of overall ratings for a selection of Amazon products. The data show a pronounced tendency towards polarized reviews, with 5-star ratings being the most common at 43.2%, followed by 1-star ratings at 28.8%. The absence of 3-star ratings is notable, indicating a potential data collection peculiarity or an intentional exclusion from the dataset.

![Figure 3.2: Frequency Distribution of Amazon Customer Reviews](/images/Figure2.png)

Figure 3.3 presents a binary classification of review sentiments where 1 and 2-star ratings are designated as negative (0), and 4 and 5-star ratings as positive (1). From the graph, it is observed that the number of positive reviews slightly exceeds negative ones, with counts of 144,240 and 150,000 respectively, indicating a moderately balanced distribution between the two sentiment categories in the dataset.

![Figure 3.3: Outcome Distribution of Amazon Customer Reviews](/images/Figure3.png)

Figure 3.4 illustrates the average word count of reviews categorized by Outcome. Negative reviews (0) have a higher average length of 11.81 words, while positive reviews (1) are shorter, with an average of 7.22 words. This suggests that users tend to write longer reviews when expressing negative sentiments.

![Figure 3.4: Analysis of Review Lengths by Outcome](/images/Figure4.png)

### 3.2 Preprocessing of Dataset
Text preprocessing is essential for improving sentiment analysis model performance. The effectiveness of these models relies heavily on the quality of input data. Therefore, preprocessing unstructured user-generated content is crucial. The steps of preprocessing are outlined as follows:

- **Tokenization.** Tokenization involves breaking down a sentence into individual words or units for feature extraction, accomplished using the tokenize module of the NLTK library.
- **Lowercasing.** Converting all tokens to lowercase is a vital preprocessing step to maintain consistency across the dataset. This uniformity is crucial for accurate feature extraction as it reduces the variability of the input data, simplifying the model's learning process.
- **Lemmatization.** Lemmatization reduces words to their base or root form, helping machines recognize different forms of the same word. This process is facilitated by the WordNet lemmatizer module in the NLTK library. Stemming on the other hand typically involves chopping off the end of the word, relying on heuristics, without any understanding of the context. For instance, the stemmer might reduce the word "flies" to "fli," which is not a valid word. Lemmatization is favored over stemming for its context-aware approach that accurately reduces words to their dictionary form, retains full semantic meaning crucial for NLP tasks, and adeptly handles irregular words, ensuring words are correctly interpreted by downstream processes (Sarkar, 2019).
- **Removing Stopwords.** Stop words, such as "I," "to," and "the," are considered insignificant for analysis and are filtered out using the NLTK stopwords corpus (Sarkar, 2019). The use of the standard NLTK stopwords list without additions maintains the integrity of contextual meaning, as expanding the list may not significantly enhance NLP model performance and could compromise computational efficiency (Sarica and Luo, 2021).
- **Removing Punctuation.** Although punctuation plays a crucial communicative role in sentences, their presence complicates processing and increases dataset size. The necessity of including punctuation varies by application (Sarkar, 2019). For topic classification, it's often less important and thus removed.

### 3.3 Vectorization
Vectorization transforms text into numerical vectors, a necessary step for machine learning classifiers to interpret and analyze language data, facilitating accurate predictions and analyses. The following Vectorization methods are used:

- **Term Frequency-Inverse Document Frequency (TF-IDF):** TF-IDF measures a word's importance in a document relative to a collection of documents, combining how often a word appears in a document (TF) with the rarity of the word across all documents (IDF). This method highlights unique words by reducing the weight of common ones. Utilizing the TfidfVectorizer and TfidfTransformer from the Scikit-learn library, TF-IDF models transform documents into numerical vectors, emphasizing single words (unigrams). The TF-IDF method is preferred over simple count vectorization techniques due to its ability to assign weights based on word importance, thereby efficiently distinguishing between common and more informative words. This characteristic is particularly useful in reducing the impact of common stop words, amplifying the relevance of more significant words in the corpus (Kozhevnikov & Pankratova, 2020).
- **Vectorization using Topic Modeling with Nonnegative Matrix Factorization (NMF):** NMF is a topic modeling approach that decomposes a corpus into topics represented by words, transforming the term-document matrix into two distinct matrices: one representing documents and the other topics. This method facilitates the classification of documents based on their topic distribution, effectively uncovering latent themes within texts. A key advantage of NMF is its restriction against negative elements, aligning with the reality that word counts in documents or topics cannot be negative (Lee & Seung, 1999). For this study, the number of estimators for NMF is set to 100, chosen to align with the 100-dimensional setting of the GloVe embeddings, ensuring a balanced comparison across the two models in terms of feature representation capability. NMF is preferred among topic modeling methods for its efficiency in extracting more distinct themes from texts (Egger & Yu, 2022), enhancing the understanding of sentiment within the corpus.
- **Vectorization with GloVe Embeddings:** GloVe (Global Vectors for Word Representation) utilizes global co-occurrence statistics to generate word embeddings, merging matrix factorization's statistical leverage with local context's insight. This method efficiently captures complex semantic and syntactic text patterns, improving performance in tasks like word analogy and similarity (Pennington, Socher, & Manning, 2014). In this study, the 100-dimension vector word embeddings (‘glove.6B.100d.txt’) were selected to ensure consistency with the NMF's 100-estimator setting, facilitating an equitable comparison in terms of dimensionality and computational efficiency. This choice is especially relevant for the customer reviews dataset, which averages fewer than 12 words per review (see Figure 3.4), making GloVe particularly suitable for handling such brief texts. GloVe is favored over alternatives such as Word2Vec, FastText, and BERT because it offers an optimal mix of understanding complex textual patterns and computational efficiency (Wang, Nulty, & Lillis, 2020).

### 3.4 Classification
The study evaluated four machine learning classifiers using 5-fold cross-validation to determine the most effective model for the dataset. The classifiers included Random Forest, Support Vector Machine (SVM), Naive Bayes, and Gradient Boosting, chosen for their wide applicability and proven performance across different types of data and classification tasks. Each classifier was implemented with its default parameters as specified in Scikit-learn, allowing for an unbiased comparison across models. Neural Networks, despite their potential for improved learning via deep learning techniques, were not considered in this analysis due to their significant computational requirements and complexity (Rakhmanov, 2020). This approach aimed to strike a balance between computational efficiency and predictive accuracy.

- **Random Forest (RF).** RF is an ensemble learning method that builds multiple decision trees to make predictions, combining their results to improve accuracy and reduce overfitting. It excels in processing complex textual data, like customer reviews, by leveraging its ability to handle diverse features and noisy data (Probst et al., 2019). This makes it well-suited for tasks such as sentiment analysis, where it can effectively manage the variability inherent in text data.
- **Support Vector Machine (SVM).** SVM is a classifier designed to find the optimal boundary (hyperplane) in high-dimensional spaces, making it ideal for binary classification. Its strength lies in managing high-dimensional data, often found in text classification, where it efficiently handles large feature spaces and non-linear data relationships (Suykens & Vandewalle, 1999). This capability renders SVM highly effective for complex text analysis tasks.
- **Naive Bayes (NB).** NB classifier is a probabilistic classifier that employs Bayes' theorem, assuming feature independence. It shines in processing categorical data and scales well with the number of features, making it efficient for text data. Its simplicity and effectiveness in modeling the probabilistic aspects of language enhance precision in applications like sentiment analysis, where it quickly processes and classifies large volumes of text (Rennie et al., 2003).
- **Gradient Boosting.** Gradient Boosting is chosen for its sequential model improvement and flexibility with loss functions, which enhances prediction accuracy for various classifications. It applies regularization to mitigate overfitting, optimizing model generalizability. Its robust performance in complex learning tasks is well-documented, asserting its efficiency and adaptability in tackling high-dimensional data (Friedman, J. H., 2001)

In the context of binary classification where predictions are binary (True or False) against actual Positive (P) and Negative (N) outcomes, performance metrics can be delineated through four outcomes: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). Consequently, Precision (PR), Recall (RC), and Accuracy (ACC) are defined by these formulas: 

![Formula](/images/Formulas.png)


Given the study's focus on evaluating the impact of text vectorization techniques on model performance, Accuracy has been selected as the primary metric. This decision is bolstered by the balanced nature of the dataset (See Figure 3.4), which mitigates the common issue of metric distortion in imbalanced datasets (Sokolova & Lapalme, 2009). Thus, Accuracy provides a straightforward and reliable measure of overall model performance across both positive and negative classifications.

## 4. Findings

![Table 3.1: Classifier Performance with Different Vectorization Methods](/images/Table1.png)

Table 3.1 provides a comparative analysis of the accuracy scores for various classification models against different vectorization methods. The RF and SVM models emerged as the top performers with accuracies of 84.31% (RF) and 85.00% (SVM) using TF-Idf vectorization, which is consistent with existing literature (Rakhmanov, 2020).

The analysis further highlights a discernible trade-off between model accuracy and computational efficiency. While RF and SVM show comparable accuracies, SVM demands significantly more CPU time, particularly evident with TF-Idf where SVM requires approximately 21.4 seconds for model training as opposed to RF's expedient 1.98 seconds. This pronounced difference in computational demand underscores RF's advantage as a more efficient yet equally accurate classifier.

The impact of vectorization type on model performance is non-uniform across different classifiers. For example, with NMF vectorization, RF's accuracy drops to 82.98%, but its CPU time decreases to 0.86 seconds, offering a faster but less accurate alternative. GloVe vectorization presents a mixed effect: it slightly reduces RF's accuracy to 79.92% but also lowers its CPU time to 1.10 seconds, suggesting a nuanced interplay between vectorization methods and classifier efficiency.

The absence of an accuracy score for Naïve Bayes (NB) with GloVe vectorization (marked as N/A) is attributed to compatibility issues; NB's reliance on discrete frequency counts is mismatched with GloVe's continuous, dense embeddings.

RF is determined to be the classifier of choice, when taking into account accuracy and CPU time. Subsequently, a grid search will be employed to refine the parameters, striving to find the RF configuration that optimally combines accuracy and processing speed.


![Table 3.2: RF Classifier Performance with Hyperparameter Tuning](/images/Table2.png)


Table 3.2 delineates RF classifier's performance following hyperparameter optimization across three vectorization techniques: TF-Idf, NMF, and GloVe. Utilizing a grid search on 12 combinations of tree depths (20, 50, 100, None) and estimators (10, 100, 250), the analysis engaged 5-fold cross-validation to enhance model validation and prevent overfitting. The aim is to pinpoint a configuration marrying high accuracy with low computational cost.

When considering accuracy alone, Model 3 (with no maximum depth and 250 estimators) delivers the highest performance for both TF-Idf and NMF vectorizations. For the GloVe vectorization, Model 9 (with a maximum depth of 50 and 250 estimators) reaches peak accuracy. Yet, when computational efficiency is considered, Model 1 (with no maximum depth and 10 estimators) emerges as the prime choice for TF-Idf vectorization, achieving high accuracy at 83.45% with a remarkably low CPU time of 0.185 seconds.

The results underscore a prevalent trend: while increasing the number of estimators and the maximum depth slightly boosts accuracy, it also leads to proportionally longer CPU times. For applications where time efficiency is a priority, Model 1 with TF-Idf vectorization is unmatched, delivering a near-top accuracy of 83.45% while requiring only 0.185 seconds of CPU time. If accuracy is the sole concern, irrespective of computational cost, Model 3 (for TF-Idf and NMF) or Model 9 (for GloVe) would be optimal.

This study demonstrates that the selection of vectorization and model parameters must be strategic, balancing the precision of output against the speed of computation, and should be catered to the particular demands of the use case. The efficiency of the TF-Idf vectorization stands out, offering an effective balance between accuracy and CPU time for the Random Forest classifier.

![Table 3.3: Classifier Performance with Different n-grams](/images/Table3.png)

Table 3.3 assesses the RF classifier's performance with Model 1 parameters and TF-Idf vectorization with varying n-gram ranges. The study finds a modest increase in accuracy when progressing from unigrams to a combination of unigrams, bigrams, and trigrams, with accuracy inching up from 83.50% to 83.85%, consistent with findings from the literature review (Tumu et al., 2020). Interestingly, the improved accuracy does not necessarily come with increased computational demand; the CPU time required slightly decreases from 0.651 seconds for the unigram model to 0.616 seconds when bigrams are included, before rising to 0.701 seconds for the full unigram-bigram-trigram model. This nuanced finding suggests that a careful selection of n-grams can yield accuracy gains without a proportional rise in CPU time, challenging the common assertion that model complexity invariably leads to higher computational costs. The decision on the n-gram configuration must therefore consider the application's unique requirements for accuracy and processing speed.

## 5. Conclusion
In conclusion, this research explored the impact of various vectorization methods on the performance of sentiment classification models, with particular focus on the Amazon Customer Review Dataset. The study underscores that while RF classifiers attain high levels of accuracy, the choice of vectorization technique—TF-Idf, NMF, or GloVe—can significantly affect both accuracy and computational efficiency. RF, when combined with TF-Idf, consistently demonstrates an effective balance, particularly in scenarios where processing time is of essence.

The results highlight the nuanced reality that augmenting the complexity of the model—through increased estimators or n-gram range—does not necessarily lead to substantial gains in accuracy and may instead result in increased computational burden. For instance, Model 1 with TF-Idf emerges as notably efficient, achieving high accuracy with minimal CPU time.

One limitation of this study is the confinement to a single dataset, which may impact the generalizability of the findings. Moreover, while the study considers computational efficiency, it does not account for potential variances in computational environments.

Future work should extend the evaluation to multiple datasets to validate the observed trends and potentially explore the integration of deep learning approaches, which, despite their computational intensity, may offer further enhancements in classification performance. The ultimate goal remains to develop a sentiment analysis tool that is both highly accurate and computationally efficient, adapting to the ever-evolving landscape of consumer feedback in the digital domain.

### References
Egger, R., & Yu, J. (2022). A topic modeling comparison between LDA, NMF, Top2Vec, and BERTopic to demystify Twitter posts. Frontiers in Sociology, 7, Article 886498. https://doi.org/10.3389/fsoc.2022.886498

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189-1232. https://doi.org/10.1214/aos/1013203451

Kozhevnikov, V. A., & Pankratova, E. S. (2020). Research of the text data vectorization and classification algorithms of machine learning. ISJ Theoretical & Applied Science, 85, 574-585. Lee, D. D., & Seung, H. S. (1999). Algorithms for non-negative matrix factorization. In Advances in Neural Information Processing Systems 13 (NIPS 2000) (pp. 556-562). Denver, CO, USA.

Nguyen, H., Veluchamy, A., Diop, M., & Iqbal, R. (2018). Comparative study of sentiment analysis with product reviews using machine learning and lexicon-based approaches. SMU Data Science Review, 1(4). https://scholar.smu.edu/datasciencereview/vol1/iss4/7

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. In Proceedings of the Empirical Methods in Natural Language Processing (EMNLP 2014) (pp. 1532-1543).

Probst, P., Wright, M. N., & Boulesteix, A.-L. (2019). Hyperparameters and tuning strategies for random forest. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 9(3), e1301. https://doi.org/10.1002/widm.1301

Rakhmanov, O. (2020). A comparative study on vectorization and classification techniques in sentiment analysis to classify student-lecturer comments. Procedia Computer Science, 178, 194–204. https://doi.org/10.1016/j.procs.2020.11.021

Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of naive Bayes text classifiers. In Proceedings of the 20th International Conference on Machine Learning (ICML-2003) (pp. 616-623). Washington, DC, USA.

Sarica, S., & Luo, J. (2021). Stopwords in technical language processing. PLOS ONE, 16(8), e0254937. https://doi.org/10.1371/journal.pone.0254937

Sarkar, D. (2019). Text analytics with Python: A practical real-world approach to gaining actionable insights from your data. Apress.

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information Processing & Management, 45(4), 427-437. https://doi.org/10.1016/j.ipm.2009.03.002

Suykens, J. A. K., & Vandewalle, J. (1999). Least squares support vector machine classifiers. Neural Processing Letters, 9(3), 293–300. https://doi.org/10.1023/A:1018628609742

Tumu, P., Manchenasetty, V., & Rege, M. (2020). Context-based sentiment analysis approach using N-Gram and word vectorization methods. Issues in Information Systems, 21(3), 59-65. https://doi.org/10.48009/3_iis_2020_59-65

Wang, C., Nulty, P., & Lillis, D. (2020). A Comparative Study on Word Embeddings in Deep Learning for Text Classification. Proceedings of NLPIR 2020, December 18-20, 2020, Seoul, Republic of Korea. ACM. https://doi.org/10.1145/3443279.3443304
