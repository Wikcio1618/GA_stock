# Genetic Algorithms Project - superficial stock market
## Encoding, environment and rules
In the project the environment consists of stock market and investors. Stock market (i.e. daily, discrete prizes of oil) is generated based on following rules:
1. There are 2 random sequences with period $T_1$
2. Those change every $T_2$ steps, resulting in global and local periodic sequence
The population is made of investors and their chromosome is of length $2^l$, where l=4 is number of previous time steps that the decision to 1-buy, 0-sell is taken. (One should already notice, that alphabet is binary). Each position in the chromosome corresponds with a different history on the market. For instance a position may correspond with such sequence of 4 previous days: drop, rise, rise drop. If shuch changes occur on the market, investors whose gene at this position was 1, will invest their money and buy the oil. Investment rate is constant and universal.
## Main project aim
Double periodic sequence on stock market means, that the invironment is changing every $T_2$, which poses a problem for population to adapt. With that in mind, Haploidal and diploidal encoding can be compared and it's result are there in *train.ipynb* file. Diploidal encoding allowed investors to make more money over larger period of time.

## Results
* Different methods of genetic algorithms are implemented and explored, namely:
    - crossover
    - mutation
    - selection
    - diploidal encoding
* Diploidal encoding allowed investors to make more money over larger period of time and it's demonstrated in *train.ipynb* file on a sufficing plot.
* *GA_stock.pptx* presentation demonstrates the results in Polish language
