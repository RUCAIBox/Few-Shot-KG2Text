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

from .chrf_scorer import ChrfScorer


class Chrf(object):
    def __init__(self, nworder=2, ncorder=6, beta=2.0):
        self.nworder = nworder
        self.ncorder = ncorder
        self.beta = beta

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        chrf_scorer = ChrfScorer(nworder=self.nworder, ncorder=self.ncorder, beta=self.beta)

        hypes = []
        refs = []
        for id in sorted(gts.keys()):
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            hypes.append(hypo[0])
            refs.append(ref)

        (totalF, averageF) = chrf_scorer.compute_score(hypes, refs)

        return [totalF, averageF], None

    def method(self):
        return "chrf"