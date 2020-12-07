# Bayesian optimisation

Experimental Bayesian optimisation code for a meeting of the [London Data
Science Journal Club](https://www.meetup.com/London-Data-Science-Journal-Club/)
(LDSJC) meetup. Uses the [GPflow](https://github.com/GPflow/GPflow) Gaussian
process and [Trieste](https://github.com/secondmind-labs/trieste) Bayesian
optimisation packages.


## Meetup

The LDSJC meetup came out of an idea to explore Distill's articles on Bayesian
optimisation and Gaussian processes. However on closer reflection these present
an idealised view of the subject so I looked for some more realistic material
to cover. This suggested a game where we try to improve upon Bayesian
optimisation.


### Agenda

- play a Bayesian optimisation guessing game (how hard can it be?)
- discuss [this tutorial paper](https://arxiv.org/abs/1807.02811) on Bayesian optimisation
- further interactive exploration of Bayesian optimisation and Gaussian processes on Distill


### Resources

- [Experimental code](https://github.com/JohnReid/BayesOptJournalClub/) for
  Bayesian optimisation
- Peter Frazier's [tutorial](https://arxiv.org/abs/1807.02811) on Bayesian
  optimisation
- Ryan Adam's [tutorial
  slides](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut8_adams_slides.pdf)
  from his University of Toronto course
- Peter Roelant's [tutorial on Gaussian
  processes](https://peterroelants.github.io/posts/gaussian-process-tutorial/)
- Distill's [interactive Gaussian
  processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
- Distill's [exploration of Bayesian
  optimisation](https://distill.pub/2020/bayesian-optimization/)
- The [GPflow](https://github.com/GPflow/GPflow) Gaussian process python
  package and example notebooks
- Secondmind Labs' [Trieste](https://github.com/secondmind-labs/trieste)
  Bayesian optimisation python package and example notebooks


## Install

Pre-requisites:
- [conda](https://docs.conda.io/en/latest/)

```bash
conda env create -f environment.yml
conda activate BayesOptJC
```


## Run

```bash
jupyter notebook "Bayesian Optimisation Guessing Game.ipynb"
```
