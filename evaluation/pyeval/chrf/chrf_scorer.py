#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 Maja Popovic

# The program is distributed under the terms
# of the GNU General Public Licence (GPL)

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Publications of results obtained through the use of original or
# modified versions of the software have to cite the authors by refering
# to the following publication:

# Maja Popović (2015).
# "chrF: character n-gram F-score for automatic MT evaluation".
# In Proceedings of the Tenth Workshop on Statistical Machine Translation (WMT15), pages 392–395
# Lisbon, Portugal, September 2015.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import math
import unicodedata
import argparse
from collections import defaultdict
import time
import string


class ChrfScorer(object):
    def __init__(self, nworder, ncorder, beta):
        self.nworder = nworder
        self.ncorder = ncorder
        self.beta = beta

    def separate_characters(self, line):
        return list(line.strip().replace(" ", ""))

    def separate_punctuation(self, line):
        words = line.strip().split()
        tokenized = []
        for w in words:
            if len(w) == 1:
                tokenized.append(w)
            else:
                lastChar = w[-1]
                firstChar = w[0]
                if lastChar in string.punctuation:
                    tokenized += [w[:-1], lastChar]
                elif firstChar in string.punctuation:
                    tokenized += [firstChar, w[1:]]
                else:
                    tokenized.append(w)

        return tokenized

    def ngram_counts(self, wordList, order):
        counts = defaultdict(lambda: defaultdict(float))
        nWords = len(wordList)
        for i in range(nWords):
            for j in range(1, order + 1):
                if i + j <= nWords:
                    ngram = tuple(wordList[i:i + j])
                    counts[j - 1][ngram] += 1

        return counts

    def ngram_matches(self, ref_ngrams, hyp_ngrams):
        matchingNgramCount = defaultdict(float)
        totalRefNgramCount = defaultdict(float)
        totalHypNgramCount = defaultdict(float)

        for order in ref_ngrams:
            for ngram in hyp_ngrams[order]:
                totalHypNgramCount[order] += hyp_ngrams[order][ngram]
            for ngram in ref_ngrams[order]:
                totalRefNgramCount[order] += ref_ngrams[order][ngram]
                if ngram in hyp_ngrams[order]:
                    matchingNgramCount[order] += min(ref_ngrams[order][ngram], hyp_ngrams[order][ngram])

        return matchingNgramCount, totalRefNgramCount, totalHypNgramCount

    def ngram_precrecf(self, matching, reflen, hyplen, beta):
        ngramPrec = defaultdict(float)
        ngramRec = defaultdict(float)
        ngramF = defaultdict(float)

        factor = beta ** 2

        for order in matching:
            if hyplen[order] > 0:
                ngramPrec[order] = matching[order] / hyplen[order]
            else:
                ngramPrec[order] = 1e-16
            if reflen[order] > 0:
                ngramRec[order] = matching[order] / reflen[order]
            else:
                ngramRec[order] = 1e-16
            denom = factor * ngramPrec[order] + ngramRec[order]
            if denom > 0:
                ngramF[order] = (1 + factor) * ngramPrec[order] * ngramRec[order] / denom
            else:
                ngramF[order] = 1e-16

        return ngramF, ngramRec, ngramPrec

    def compute_score(self, hypes, refs):

        norder = float(self.nworder + self.ncorder)

        # initialisation of document level scores
        totalMatchingCount = defaultdict(float)
        totalRefCount = defaultdict(float)
        totalHypCount = defaultdict(float)
        totalChrMatchingCount = defaultdict(float)
        totalChrRefCount = defaultdict(float)
        totalChrHypCount = defaultdict(float)
        averageTotalF = 0.0

        nsent = 0
        for hline, rline in zip(hypes, refs):
            nsent += 1

            maxF = 0.0

            hypNgramCounts = self.ngram_counts(self.separate_punctuation(hline), self.nworder)
            hypChrNgramCounts = self.ngram_counts(self.separate_characters(hline), self.ncorder)

            for ref in rline:
                refNgramCounts = self.ngram_counts(self.separate_punctuation(ref), self.nworder)
                refChrNgramCounts = self.ngram_counts(self.separate_characters(ref), self.ncorder)

                # number of overlapping n-grams, total number of ref n-grams, total number of hyp n-grams
                matchingNgramCounts, totalRefNgramCount, totalHypNgramCount = self.ngram_matches(refNgramCounts,
                                                                                            hypNgramCounts)
                matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount = self.ngram_matches(refChrNgramCounts,
                                                                                                     hypChrNgramCounts)

                # n-gram f-scores, recalls and precisions
                ngramF, ngramRec, ngramPrec = self.ngram_precrecf(matchingNgramCounts, totalRefNgramCount,
                                                                totalHypNgramCount,
                                                                self.beta)
                chrNgramF, chrNgramRec, chrNgramPrec = self.ngram_precrecf(matchingChrNgramCounts, totalChrRefNgramCount,
                                                                      totalChrHypNgramCount, self.beta)

                sentF = (sum(chrNgramF.values()) + sum(ngramF.values())) / norder

                if sentF > maxF:
                    maxF = sentF
                    bestMatchingCount = matchingNgramCounts
                    bestRefCount = totalRefNgramCount
                    bestHypCount = totalHypNgramCount
                    bestChrMatchingCount = matchingChrNgramCounts
                    bestChrRefCount = totalChrRefNgramCount
                    bestChrHypCount = totalChrHypNgramCount

            # collect document level ngram counts
            for order in range(self.nworder):
                totalMatchingCount[order] += bestMatchingCount[order]
                totalRefCount[order] += bestRefCount[order]
                totalHypCount[order] += bestHypCount[order]
            for order in range(self.ncorder):
                totalChrMatchingCount[order] += bestChrMatchingCount[order]
                totalChrRefCount[order] += bestChrRefCount[order]
                totalChrHypCount[order] += bestChrHypCount[order]

            averageTotalF += maxF

        # all sentences are done

        # total precision, recall and F (aritmetic mean of all ngrams)
        totalNgramF, totalNgramRec, totalNgramPrec = self.ngram_precrecf(totalMatchingCount, totalRefCount, totalHypCount,
                                                                    self.beta)
        totalChrNgramF, totalChrNgramRec, totalChrNgramPrec = self.ngram_precrecf(totalChrMatchingCount, totalChrRefCount,
                                                                             totalChrHypCount, self.beta)

        totalF = (sum(totalChrNgramF.values()) + sum(totalNgramF.values())) / norder
        averageTotalF = averageTotalF / nsent

        return totalF, averageTotalF
