# USA-NNK
# INTRODUCTION OF COVID-NEWS-US-NNK
For citeation : Sadman, N. (2020). INTRODUCTION OF COVID-NEWS-US-NNK AND COVID-NEWS-BD-NNK DATASET. doi:10.13140/RG.2.2.19443.96802
# COVID-NEWS-BD-NNKDATASET

To collaborate  
 USA: https://docs.google.com/forms/d/1dYvEZSRIygNq3oYxLMjO4_ptrx4j3CkUSaSEOsxrFJE/edit?usp=sharing
 
and 

BD: https://docs.google.com/forms/d/1gxIhyRj2kA0LVImOQYq307qdPflUy4EO2CSrtZnznJU/edit?usp=sharing}

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4047648.svg)](https://doi.org/10.5281/zenodo.4047648)

Moderated by

Nafiz Sadman

Nishat Anjum

Kishor Datta Gupta : www.github.com/kishordgupta 




## ABSTRACT


In this paper, we introduce a collection of 1000 news report dataset on COVID-19 from two different
countries and used Natural Language Processing techniques to extract knowledge about the virus,
including the number of COVID-cases, trending topics month, sentiment analysis, etc. Moreover,
we compared how the virus spreads and impacts a developed country and a developing country. Our
curated dataset can be used in various socio-economical studies to understand news media’s effect on
public awareness.

KeywordsCovid-19·News report·Supervised Dataset·natural language processing

## 1 Introduction

There are several works based on Natural Language Processing on newspaper reports. Mining opinions from headlines
[ 1 ] using Standford NLP and SVM by Rameshbhaiet. Al.compared several algorithms on a small and large dataset.
Rubinet. al., in their paper [ 2 ], created a mechanism to differentiate fake news from real ones by building a set of
characteristics of news according to their types. The purpose was to contribute to the low resource data available for
training machine learning algorithms. Doumitet. al.in [ 3 ] have implemented LDA, a topic modeling approach to study
bias present in online news media.

However, there are not many NLP research invested in studying COVID-19. Most applications include classification of
chest X-rays and CT-scans to detect presence of pneumonia in lungs [ 4 ], a consequence of the virus. Other research
areas include studying the genome sequence of the virus[ 5 ][ 6 ][ 7 ] and replicating its structure to fight and find a vaccine.
This research is crucial in battling the pandemic. The few NLP based research publications are sentiment classification
of online tweets by Samuel et el [ 8 ] to understand fear persisting in people due to the virus. Similar work has been done
using the LSTM network to classify sentiments from online discussion forums by Jelodaret. al.[ 9 ]. NKK dataset is
the first study on a comparatively larger dataset of a newspaper report on COVID-19, which contributed to the virus’s
awareness to the best of our knowledge.

## 2 Data-set Introduction

2.1 Data Collection

We accumulated 1000 online newspaper report from United States of America (USA) on COVID-19. The newspaper
includes The Washington Post (USA) and StarTribune (USA). We have named it as “Covid-News-USA-NNK”. We also
accumulated 50 online newspaper report from Bangladesh on the issue and named it “Covid-News-BD-NNK”. The
newspaper includes The Daily Star (BD) and Prothom Alo (BD). All these newspapers are from the top provider and
top read in the respective countries. The collection was done manually by 10 human data-collectors of age group 23-
with university degrees. This approach was suitable compared to automation to ensure the news were highly relevant to
the subject. The newspaper online sites had dynamic content with advertisements in no particular order. Therefore there
were high chances of online scrappers to collect inaccurate news reports. One of the challenges while collecting the
data is the requirement of subscription. Each newspaper required $1 per subscriptions. Some criteria in collecting the
news reports provided as guideline to the human data-collectors were as follows:

- The headline must have one or more words directly or indirectly related to COVID-19.
- The content of each news must have 5 or more keywords directly or indirectly related to COVID-19.
- The genre of the news can be anything as long as it is relevant to the topic. Political, social, economical genres
    are to be more prioritized.
- Avoid taking duplicate reports.
- Maintain a time frame for the above mentioned newspapers.

To collect these data we used a google form for USA and BD. We have two human editor to go through each entry to
check any spam or troll entry.

2.2 Data Pre-processing and Statistics

Some pre-processing steps performed on the newspaper report dataset are as follows:

- Remove hyperlinks.
- Remove non-English alphanumeric characters.
- Remove stop words.
- Lemmatize text.

While more pre-processing could have been applied, we tried to keep the data as much unchanged as possible since
changing sentence structures could result us in valuable information loss. While this was done with help of a script, we
also assigned same human collectors to cross check for any presence of the above mentioned criteria.

The primary data statistics of the two dataset are shown in Table 1 and 2.

```
Table 1: Covid-News-USA-NNK data statistics
```
```
No of words per
headline
```
```
7 to 20
```
```
No of words per body
content
```
```
150 to 2100
```
```
Table 2: Covid-News-BD-NNK data statistics
No of words per
headline
```
```
10 to 20
```
```
No of words per body
content
```
```
100 to 1500
```
2.3 Dataset Repository

We used GitHub as our primary data repository in account name NKK^1. Here, we created two repositories USA-NKK^2
and BD-NNK^3. The dataset is available in both CSV and JSON format. We are regularly updating the CSV files and
regenerating JSON using a py script. We provided a python script file for essential operation. We welcome all outside
collaboration to enrich the dataset.

![Image of Yaktocat](https://github.com/NNK-Dataset/BD-NNK/blob/master/wc.png)

## 3 Literature Review


Natural Language Processing (NLP) deals with text (also known as categorical) data in computer science, utilizing
numerous diverse methods like one-hot encoding, word embedding, etc., that transform text to machine language, which
can be fed to multiple machine learning and deep learning algorithms.

Some well-known applications of NLP includes fraud detection on online media sites[ 10 ], using authorship attribution
in fallback authentication systems[ 11 ], intelligent conversational agents or chatbots[ 12 ] and machine translations used
by Google Translate[ 13 ]. While these are all downstream tasks, several exciting developments have been made in
the algorithm solely for Natural Language Processing tasks. The two most trending ones are BERT[ 14 ], which uses
bidirectional encoder-decoder architecture to create the transformer model, that can do near-perfect classification tasks
and next-word predictions for next generations, and GPT-3 models released by OpenAI[ 15 ] that can generate texts
almost human-like. However, these are all pre-trained models since they carry huge computation cost.
Information Extraction is a generalized concept of retrieving information from a dataset. Information extraction from
an image could be retrieving vital feature spaces or targeted portions of an image; information extraction from speech
could be retrieving information about names, places, etc[ 16 ]. Information extraction in texts could be identifying named
entities and locations or essential data.
Topic modeling is a sub-task of NLP and also a process of information extraction. It clusters words and phrases of the
same context together into groups. Topic modeling is an unsupervised learning method that gives us a brief idea about a
set of text. One commonly used topic modeling is Latent Dirichlet Allocation or LDA[17].

Keyword extraction is a process of information extraction and sub-task of NLP to extract essential words and phrases
from a text. TextRank [ 18 ] is an efficient keyword extraction technique that uses graphs to calculate the weight of each
word and pick the words with more weight to it.

Word clouds are a great visualization technique to understand the overall ’talk of the topic’. The clustered words give us
a quick understanding of the content.



![Image of Yaktocat](https://github.com/NNK-Dataset/BD-NNK/blob/master/da.png)
## 4 Our experiments and Result analysis


We used the wordcloud library^4 to create the word clouds. Figure 1 and 3 presents the word cloud of Covid-News-USA-
NNK dataset by month from February to May.
From the figures 1,2,3, we can point few information:

- In February, both the news paper have talked about China and source of the outbreak.
- StarTribune emphasized on Minnesota as the most concerned state. In April, it seemed to have been concerned
    more.
- Both the newspaper talked about the virus impacting the economy, i.e, bank, elections, administrations,
    markets.
- Washington Post discussed global issues more than StarTribune.
- StarTribune in February mentioned the first precautionary measurement: wearing masks, and the uncontrollable
    spread of the virus throughout the nation.
- While both the newspaper mentioned the outbreak in China in February, the weight of the spread in the United
    States are more highlighted through out March till May, displaying the critical impact caused by the virus.

We used a script to extract all numbers related to certain keywords like ’Deaths’, ’Infected’, ’Died’ , ’Infections’,
’Quarantined’, Lock-down’, ’Diagnosed’ etc from the news reports and created a number of cases for both the newspaper.
Figure 4 shows the statistics of this series.
From this extraction technique, we can observe that April was the peak month for the covid cases as it gradually rose
from February. Both the newspaper clearly shows us that the rise in covid cases from February to March was slower
than the rise from March to April. This is an important indicator of possible recklessness in preparations to battle the
virus. However, the steep fall from April to May also shows the positive response against the attack.
We used Vader Sentiment Analysis to extract sentiment of the headlines and the body. On average, the sentiments were
from -0.5 to -0.9. Vader Sentiment scale ranges from -1(highly negative to 1(highly positive). There were some cases




where the sentiment scores of the headline and body contradicted each other,i.e., the sentiment of the headline was
negative but the sentiment of the body was slightly positive. Overall, sentiment analysis can assist us sort the most
concerning (most negative) news from the positive ones, from which we can learn more about the indicators related to
COVID-19 and the serious impact caused by it. Moreover, sentiment analysis can also provide us information about
how a state or country is reacting to the pandemic.
We used PageRank algorithm to extract keywords from headlines as well as the body content. PageRank efficiently
highlights important relevant keywords in the text. Some frequently occurring important keywords extracted from both
the datasets are: ’China’, Government’, ’Masks’, ’Economy’, ’Crisis’, ’Theft’ , ’Stock market’ , ’Jobs’ , ’Election’,
’Missteps’, ’Health’, ’Response’. Keywords extraction acts as a filter allowing quick searches for indicators in case of locating situations of the economy,
how states are defending against the pandemic, the condition of the health care system etc.

## 5 Conclusion


This dataset can demonstrate how news reports could speculate the situation differently based on the news source. The
different types of experiments are possible to assert the importance of Natural Language Processing in newspaper report
analysis. We are looking for more collaborators in GitHub to enrich the dataset, which will make it possible to run
extensive deep learning experiments.

## References


[1] Chaudhary Jashubhai Rameshbhai and Joy Paulose. Opinion mining on newspaper headlines using svm and nlp.
International Journal of Electrical & Computer Engineering (2088-8708), 9(3), 2019.
[2]Victoria L Rubin, Yimin Chen, and Nadia K Conroy. Deception detection for news: three types of fakes.
Proceedings of the Association for Information Science and Technology, 52(1):1–4, 2015.
[3]Sarjoun Doumit and Ali Minai. Online news media bias analysis using an lda-nlp approach. InInternational
Conference on Complex Systems, 2011.
[4]Md Manjurul Ahsan, Kishor Datta Gupta, Mohammad Maminur Islam, Sajib Sen, Md Rahman, Moham-
mad Shakhawat Hossain, et al. Study of different deep learning approach with explainable ai for screening
patients with covid-19 symptoms: Using ct scan and chest x-ray image dataset.arXiv preprint arXiv:2007.12525,
2020.
[5]Gurjit S Randhawa, Maximillian PM Soltysiak, Hadi El Roz, Camila PE de Souza, Kathleen A Hill, and Lila Kari.
Machine learning using intrinsic genomic signatures for rapid classification of novel pathogens: Covid-19 case
study.Plos one, 15(4):e0232391, 2020.

[6]Ahmad Alimadadi, Sachin Aryal, Ishan Manandhar, Patricia B Munroe, Bina Joe, and Xi Cheng. Artificial
intelligence and machine learning to fight covid-19, 2020.
[7]Shreshth Tuli, Shikhar Tuli, Rakesh Tuli, and Sukhpal Singh Gill. Predicting the growth and trend of covid-
pandemic using machine learning and cloud computing.Internet of Things, page 100222, 2020.
[8]Jim Samuel, GG Ali, Md Rahman, Ek Esawi, Yana Samuel, et al. Covid-19 public sentiment insights and machine
learning for tweets classification.Information, 11(6):314, 2020.
[9] Hamed Jelodar, Yongli Wang, Rita Orji, and Hucheng Huang. Deep sentiment classification and topic discovery
on novel coronavirus or covid-19 online discussions: Nlp using lstm recurrent neural network approach.arXiv
preprint arXiv:2004.11695, 2020.

[10]Nafiz Sadman, Kishor Datta Gupta, Ariful Haque, Subash Poudyal, and Sajib Sen. Detect review manipulation by
leveraging reviewer historical stylometrics in amazon, yelp, facebook and google reviews. InProceedings of the
2020 The 6th International Conference on E-Business and Applications, pages 42–47, 2020.

[11]Nafiz Sadman, Kishor Datta Gupta, Ariful Haque, Subash Poudyal, and Sajib Sen. Stylometry as a reliable
method for fallback authentication. InProceedings of the 2020 17th International Conference on Electrical
Engineering/Electronics, Computer, Telecommunications and Information Technology, 2020.

[12]Ethan Fast, Binbin Chen, Julia Mendelsohn, Jonathan Bassen, and Michael S Bernstein. Iris: A conversational
agent for complex tasks. InProceedings of the 2018 CHI Conference on Human Factors in Computing Systems,
pages 1–12, 2018.

[13] Philipp Koehn.Statistical machine translation. Cambridge University Press, 2009.

[14]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional
transformers for language understanding.arXiv preprint arXiv:1810.04805, 2018.

[15]Will Douglas Heavenarchive page. Openai’s new language generator gpt-3 is shockingly good—and completely
mindless https://www.technologyreview.com/2020/07/20/1005454/openai-machine-learning-language-generator-
gpt-3-nlp/. Technical report.

[16]Chin-Hui Lee and Sabato Marco Siniscalchi. An information-extraction approach to speech processing: Analysis,
detection, verification, and recognition.Proceedings of the IEEE, 101(5):1089–1115, 2013.

[17]Hamed Jelodar, Yongli Wang, Chi Yuan, Xia Feng, Xiahui Jiang, Yanchao Li, and Liang Zhao. Latent dirich-
let allocation (lda) and topic modeling: models, applications, a survey. Multimedia Tools and Applications,
78(11):15169–15211, 2019.

[18]Monica Bianchini, Marco Gori, and Franco Scarselli. Inside pagerank.ACM Transactions on Internet Technology
(TOIT), 5(1):92–128, 2005.


