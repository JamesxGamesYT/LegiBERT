import os
import shutil
import sys
from tempfile import TemporaryDirectory
import torch
import json

nlp_path = os.path.abspath("../../")
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

from utils_nlp.models.transformers.abstractive_summarization_bertsum import (
    BertSumAbs,
    BertSumAbsProcessor,
)

from utils_nlp.dataset.cnndm import CNNDMSummarizationDataset
from utils_nlp.eval import compute_rouge_python

from utils_nlp.models.transformers.datasets import SummarizationDataset
import nltk
from nltk import tokenize

import pandas as pd
import pprint
import scrapbook as sb

# the data path used to save the downloaded data file
DATA_PATH = TemporaryDirectory().name
# DATA_PATH = "/tmp/tmp9zp4jblv/"
# DATA_PATH = "/tmp/tmp5t0vl9gm"
# The number of lines at the head of data file used for preprocessing. -1 means all the lines.
TOP_N = 100
# TOP_N = 1

train_dataset, test_dataset = CNNDMSummarizationDataset(
    top_n=TOP_N, local_cache_path=DATA_PATH, prepare_extractive=False
)
print(len(train_dataset))
print(len(test_dataset))

# notebook parameters
# the cache path
CACHE_PATH = TemporaryDirectory().name
# CACHE_PATH = "/tmp/tmp9zp4jblv/"
# CACHE_PATH = "/tmp/tmp5t0vl9gm/fine_tuned"
# CACHE_PATH = "/tmp/tmp5t0vl9gm"

# model parameters
MODEL_NAME = "bert-base-uncased"
# MAX_POS = 768
MAX_POS = 512
MAX_SOURCE_SEQ_LENGTH = 640
MAX_TARGET_SEQ_LENGTH = 140

# mixed precision setting. To enable mixed precision training, follow instructions in SETUP.md.
FP16 = False
if FP16:
    FP16_OPT_LEVEL = "O2"

# fine-tuning parameters
# batch size, unit is the number of tokens
BATCH_SIZE_PER_GPU = 1


# GPU used for training
NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS > 0:
    BATCH_SIZE = NUM_GPUS * BATCH_SIZE_PER_GPU
else:
    BATCH_SIZE = 1


# Learning rate
LEARNING_RATE_BERT = 5e-4 / 2.0
LEARNING_RATE_DEC = 0.05 / 2.0


# How often the statistics reports show up in training, unit is step.
REPORT_EVERY = 10
SAVE_EVERY = 500

# total number of steps for training
# MAX_STEPS = 1e3
MAX_STEPS = 5e3

# # if not QUICK_RUN:
#     # MAX_STEPS = 5e3

WARMUP_STEPS_BERT = 2000
WARMUP_STEPS_DEC = 1000
# # processor which contains the colloate function to load the preprocessed data
processor = BertSumAbsProcessor(cache_dir=CACHE_PATH, max_src_len=MAX_SOURCE_SEQ_LENGTH, max_tgt_len=MAX_TARGET_SEQ_LENGTH)
# # summarizer
summarizer = BertSumAbs(
    processor, cache_dir=CACHE_PATH, max_pos_length=MAX_POS
)
BATCH_SIZE_PER_GPU*NUM_GPUS
summarizer.fit(
    train_dataset,
    num_gpus=NUM_GPUS,
    batch_size=BATCH_SIZE,
    max_steps=MAX_STEPS,
    learning_rate_bert=LEARNING_RATE_BERT,
    learning_rate_dec=LEARNING_RATE_DEC,
    warmup_steps_bert=WARMUP_STEPS_BERT,
    warmup_steps_dec=WARMUP_STEPS_DEC,
    save_every=SAVE_EVERY,
    report_every=REPORT_EVERY * 5,
    fp16=FP16,
    # checkpoint="saved checkpoint path"
)
summarizer.save_model(MAX_STEPS, os.path.join(CACHE_PATH, "bertsumabs.pt"))
checkpoint = torch.load(os.path.join(CACHE_PATH, "bertsumabs.pt"), map_location="cpu")
summarizer = BertSumAbs(
    processor, cache_dir=CACHE_PATH, max_pos_length=MAX_POS, test=True
)
summarizer.model.load_checkpoint(checkpoint['model'])
TEST_TOP_N = 32
# if not QUICK_RUN:
    # TEST_TOP_N = len(test_dataset)

if NUM_GPUS:
    BATCH_SIZE = NUM_GPUS * BATCH_SIZE_PER_GPU
else:
    BATCH_SIZE = 1
    
shortened_dataset = test_dataset.shorten(top_n=TEST_TOP_N)
src = shortened_dataset.get_source()
reference_summaries = [" ".join(t).rstrip("\n") for t in shortened_dataset.get_target()]
generated_summaries = summarizer.predict(
    shortened_dataset, batch_size=BATCH_SIZE, num_gpus=NUM_GPUS
)
# assert len(generated_summaries) == len(reference_summaries)
print(len(generated_summaries), len(reference_summaries), "generated summaries, reference summaries")
print(src[0])
print(generated_summaries[0])
print(reference_summaries[0])

rouge_scores = compute_rouge_python(cand=generated_summaries, ref=reference_summaries)
pprint.pprint(rouge_scores)
with open("src.txt", "w") as f:
    f.write(str(src))
with open("generated_summaries.txt", "w") as f:
    f.write(str(generated_summaries))
with open("reference_summaries.txt", "w") as f:
    f.write(str(reference_summaries))
# for testing
# sb.glue("rouge_2_f_score", rouge_scores['rouge-2']['f'])
# sb.glue("rouge_2_f_score", rouge_scores[1]['f'])

source = """
But under the new rule, set to be announced in the next 48 hours, Border Patrol agents would immediately return anyone to Mexico — without any detainment and without any due process — who attempts to cross the southwestern border between the legal ports of entry. The person would not be held for any length of time in an American facility.

Although they advised that details could change before the announcement, administration officials said the measure was needed to avert what they fear could be a systemwide outbreak of the coronavirus inside detention facilities along the border. Such an outbreak could spread quickly through the immigrant population and could infect large numbers of Border Patrol agents, leaving the southwestern border defenses weakened, the officials argued.
The Trump administration plans to immediately turn back all asylum seekers and other foreigners attempting to enter the United States from Mexico illegally, saying the nation cannot risk allowing the coronavirus to spread through detention facilities and Border Patrol agents, four administration officials said.
The administration officials said the ports of entry would remain open to American citizens, green-card holders and foreigners with proper documentation. Some foreigners would be blocked, including Europeans currently subject to earlier travel restrictions imposed by the administration. The points of entry will also be open to commercial traffic."""
source="""
"SECTION 1. SHORT TITLE.\n\n    This Act may be cited as the ``National Science Education Tax \nIncentive for Businesses Act of 2007''.\n\nSEC. 2. CREDITS FOR CERTAIN CONTRIBUTIONS BENEFITING SCIENCE, \n              TECHNOLOGY, ENGINEERING, AND MATHEMATICS EDUCATION AT THE \n              ELEMENTARY AND SECONDARY SCHOOL LEVEL.\n\n    (a) In General.--Subpart D of part IV of subchapter A of chapter 1 \nof the Internal Revenue Code of 1986 (relating to business related \ncredits) is amended by adding at the end the following new section:\n\n``SEC. 45O. CONTRIBUTIONS BENEFITING SCIENCE, TECHNOLOGY, ENGINEERING, \n              AND MATHEMATICS EDUCATION AT THE ELEMENTARY AND SECONDARY \n              SCHOOL LEVEL.\n\n    ``(a) In General.--For purposes of section 38, the elementary and \nsecondary science, technology, engineering, and mathematics (STEM) \ncontributions credit determined under this section for the taxable year \nis an amount equal to 100 percent of the qualified STEM contributions \nof the taxpayer for such taxable year.\n    ``(b) Qualified STEM Contributions.--For purposes of this section, \nthe term `qualified STEM contributions' means--\n            ``(1) STEM school contributions,\n            ``(2) STEM teacher externship expenses, and\n            ``(3) STEM teacher training expenses.\n    ``(c) STEM School Contributions.--For purposes of this section--\n            ``(1) In general.--The term `STEM school contributions' \n        means--\n                    ``(A) STEM property contributions, and\n                    ``(B) STEM service contributions.\n            ``(2) STEM property contributions.--The term `STEM property \n        contributions' means the amount which would (but for subsection \n        (f)) be allowed as a deduction under section 170 for a \n        charitable contribution of STEM inventory property if--\n                    ``(A) the donee is an elementary or secondary \n                school described in section 170(b)(1)(A)(ii),\n                    ``(B) substantially all of the use of the property \n                by the donee is within the United States or within the \n                defense dependents' education system for educational \n                purposes in any of the grades K-12 that are related to \n                the purpose or function of the donee,\n                    ``(C) the original use of the property begins with \n                the donee,\n                    ``(D) the property will fit productively into the \n                donee's education plan,\n                    ``(E) the property is not transferred by the donee \n                in exchange for money, other property, or services, \n                except for shipping, installation and transfer costs, \n                and\n                    ``(F) the donee's use and disposition of the \n                property will be in accordance with the provisions of \n                subparagraphs (B) and (E).\n        The determination of the amount of deduction under section 170 \n        for purposes of this paragraph shall be made as if the \n        limitation under section 170(e)(3)(B) applied to all STEM \n        inventory property.\n            ``(3) STEM service contributions.--The term `STEM service \n        contributions' means the amount paid or incurred during the \n        taxable year for STEM services provided in the United States or \n        in the defense dependents' education system for the exclusive \n        benefit of students at an elementary or secondary school \n        described in section 170(b)(1)(A)(ii) but only if--\n                    ``(A) the taxpayer is engaged in the trade or \n                business of providing such services on a commercial \n                basis, and\n                    ``(B) no charge is imposed for providing such \n                services.\n            ``(4) STEM inventory property.--The term `STEM inventory \n        property' means, with respect to any contribution to a school, \n        any property--\n                    ``(A) which is described in paragraph (1) or (2) of \n                section 1221(a) with respect to the donor, and\n                    ``(B) which is determined by the school to be \n                needed by the school in providing education in grades \n                K-12 in the areas of science, technology, engineering, \n                or mathematics.\n            ``(5) STEM services.--The term `STEM services' means, with \n        respect to any contribution to a school, any service determined \n        by the school to be needed by the school in providing education \n        in grades K-12 in the areas of science, technology, \n        engineering, or mathematics, including teaching courses of \n        instruction at such school in any such area.\n            ``(6) Defense dependents' education system.--For purposes \n        of this subsection, the term `defense dependents' education \n        system' means the program established and operated under the \n        Defense Dependents' Education Act of 1978 (20 U.S.C. 921 et \n        seq.).\n    ``(d) STEM Teacher Externship Expenses.--For purposes of this \nsection--\n            ``(1) In general.--The term `STEM teacher externship \n        expenses' means any amount paid or incurred to carry out a STEM \n        externship program of the taxpayer but only to the extent that \n        such amount is attributable to the participation in such \n        program of any eligible STEM teacher, including amounts paid to \n        such a teacher as a stipend while participating in such \n        program.\n            ``(2) STEM externship program.--The term `STEM externship \n        program' means any program--\n                    ``(A) established by a taxpayer engaged in a trade \n                or business within an area of science, technology, \n                engineering, or mathematics, and\n                    ``(B) under which eligible STEM teachers receive \n                training to enhance their teaching skills in the areas \n                of science, technology, engineering, or mathematics or \n                otherwise improve their knowledge in such areas.\n            ``(3) Eligible stem teacher.--The term `eligible STEM \n        teacher' means any individual--\n                    ``(A) who is a teacher in grades K-12 at an \n                educational organization described in section \n                170(b)(1)(A)(ii) which is located in the United States \n                or which is located on a United States military base \n                outside the United States, and\n                    ``(B) whose teaching responsibilities at such \n                school include, or are likely to include, any course in \n                the areas of science, technology, engineering, or \n                mathematics.\n    ``(e) STEM Teacher Training Expenses.--The term `STEM teacher \ntraining expenses' means any amount paid or incurred by a taxpayer \nengaged in a trade or business within an area of science, technology, \nengineering, or mathematics which is attributable to the participation \nof any eligible STEM teacher in a regular training program provided to \nemployees of the taxpayer which is determined by such teacher's school \nas enhancing such teacher's teaching skills in the areas of science, \ntechnology, engineering, or mathematics.\n    ``(f) Denial of Double Benefit.--No deduction shall be allowed \nunder this chapter for any amount allowed as a credit under this \nsection.''.\n    (b) Conforming Amendments.--\n            (1) Section 38(b) of such Code is amended by striking \n        ``plus'' at the end of paragraph (30), by striking the period \n        at the end of paragraph (31), and inserting ``, plus'', and by \n        adding at the end the following new paragraph:\n            ``(32) the elementary and secondary science, technology, \n        engineering, and mathematics (STEM) contributions credit \n        determined under section 45O.''.\n            (2) The table of sections for subpart D of part IV of \n        subchapter A of chapter 1 of such Code is amended by adding at \n        the end the following new item:\n\n``Sec. 45O. Contributions benefiting science, technology, engineering, \n                            and mathematics education at the elementary \n                            and secondary school level.''.\n    (c) Effective Date.--The amendments made by this section shall \napply to taxable years beginning after the date of the enactment of \nthis Act.","summary":"National Science Education Tax Incentive for Businesses Act of 2007 - Amends the Internal Revenue Code to allow a general business tax credit for contributions of property or services to elementary and secondary schools and for teacher training to promote instruction in science, technology, engineering, or mathematics .","title":"To amend the Internal Revenue Code of 1986 to encourage businesses to improve math and science education at elementary and secondary schools.","text_len":8494,"sum_len":321}
{"bill_id":"112_hr2873","text":"SECTION 1. SHORT TITLE.\n\n    This Act may be cited as the ``Small Business Expansion and Hiring \nAct of 2011''.\n\nSEC. 2. BUSINESS CREDIT FOR RETENTION OF CERTAIN INDIVIDUALS NEWLY \n              HIRED BEFORE 2013.\n\n    (a) In General.--Subpart D of part IV of subchapter A of chapter 1 \nof the Internal Revenue Code of 1986 (relating to business-related \ncredits) is amended by adding at the end the following new section:\n\n``SEC. 45S. RETENTION OF CERTAIN INDIVIDUALS NEWLY HIRED BEFORE 2013.\n\n    ``(a) In General.--For purposes of section 38, in the case of any \ntaxable year ending after the date of the enactment of this section and \nbeginning before January 1, 2013, the retained worker credit determined \nunder this section for the taxable year is the aggregate of the lesser \nof--\n            ``(1) $4,000 ($6,000 in the case of a long-term unemployed \n        individual), or\n            ``(2) 6.2 percent of the wages (as defined in section \n        3401(a)) paid by the taxpayer to such retained worker during \n        the 52 consecutive week period referred to in subsection \n        (c)(2).\n    ``(b) Limitations.--\n            ``(1) Increase in employment.--The number of retained \n        workers taken into account under subsection (a) shall not \n        exceed the excess of (if any)--\n                    ``(A) the number of employees of the taxpayer at \n                the end of the taxable year, over\n                    ``(B) the number of employees of the taxpayer at \n                the beginning of the taxable year.\n            ``(2) Dollar limitation.--The amount allowed as a credit \n        under subsection (a) for a taxable year with respect to any \n        business location of the employer shall not exceed $400,000.\n            ``(3) Special rules.--\n                    ``(A) Business-location specific.--All \n                determinations under this section regarding the number \n                of employees shall be determined on a location basis.\n                    ``(B) Employees rotated among business not \n                eligible.--An employee who is moved from one location \n                of the taxpayer to another location shall not be taken \n                into account for purposes of paragraph (1).\n    ``(c) Definitions.--For purposes of this section--\n            ``(1) Retained worker.--The term `retained worker' means \n        any qualified individual--\n                    ``(A) who was employed by the taxpayer on any date \n                during the taxable year,\n                    ``(B) who was so employed by the taxpayer for a \n                period of not less than 52 consecutive weeks, and\n                    ``(C) whose wages (as defined in section 3401(a)) \n                for such employment during the last 26 weeks of such \n                period equaled at least 80 percent of such wages for \n                the first 26 weeks of such period.\n            ``(2) Qualified individual.--The term `qualified \n        individual' means any individual who--\n                    ``(A) begins employment with a qualified employer \n                after December 31, 2010, and before January 1, 2014,\n                    ``(B) certifies by signed affidavit, under \n                penalties of perjury, that such individual has not been \n                employed for 40 hours or more per week during the 60-\n                day period ending on the date such individual begins \n                such employment,\n                    ``(C) is not employed by the qualified employer to \n                replace another employee of such employer unless such \n                other employee separated from employment voluntarily or \n                for cause, and\n                    ``(D) is not an individual described in section \n                51(i)(1) (applied by substituting `qualified employer' \n                for `taxpayer' each place it appears).\n            ``(3) Qualified employer.--\n                    ``(A) In general.--The term `qualified employer' \n                means any employer other than the United States, any \n                State, or any political subdivision thereof, or any \n                instrumentality of the foregoing which employed an \n                average of less than 100 employees on business days \n                during such taxable year.\n                    ``(B) Treatment of employees of post-secondary \n                educational institutions.--Notwithstanding subparagraph \n                (A), the term `qualified employer' includes any \n                employer which is a public institution of higher \n                education (as defined in section 101(b) of the Higher \n                Education Act of 1965).\n            ``(4) Long-term unemployed individual.--The term `long-term \n        unemployed individual' means an individual who was in receipt \n        of unemployment compensation under State or Federal law for not \n        less than 26 weeks during the 1-year period ending on the day \n        the individual is hired by the employer.''.\n    (b) Credit Allowed as Business Credit.--Section 38(b) of the \nInternal Revenue Code of 1986 (relating to current year business \ncredit) is amended by striking ``plus'' at the end of paragraph (35), \nby striking the period at the end of paragraph (36) and inserting ``, \nplus'', and by adding at the end the following new paragraph:\n            ``(37) the retained worker credit determined under section \n        45S.''.\n    (c) Limitation on Carryforward.--Section 39(a) of such Code is \namended by adding at the end the following:\n            ``(5) 3-year carryforward for retained worker credit.--In \n        the case of the retained worker credit, paragraph (2) shall be \n        applied--\n                    ``(A) by substituting `3 taxable years' for `21 \n                taxable years' in subparagraph (A) thereof, and\n                    ``(B) by substituting `2 taxable years' for `20 \n                taxable years' in subparagraph (B) thereof.''.\n    (d) Clerical Amendment.--The table of sections for subpart D of \npart IV of subchapter A of chapter 1 of the Internal Revenue Code of \n1986 is amended by inserting after the item relating to section 45R the \nfollowing new item:\n\n``Sec. 45S. Retention of certain individuals newly hired before \n                            2013.''.\n    (e) Effective Date.--The amendments made by this section shall \napply to taxable years beginning after the date of the enactment of \nthis Act.","summary":"Small Business Expansion and Hiring Act of 2011 - Amends the Internal Revenue Code to allow nongovernmental employers who employ an average of fewer than 100 employees during a taxable year a retained worker tax credit until December 31, 2012, for the lesser of $4,000 or 6.2 of the wages paid to a retained worker during a period of not less than 52 consecutive weeks of employment. Limits the amount of such credit with respect to any business location of the employer to $400,000 and provides that the number of retained workers taken into account for such credit shall not exceed the excess of the number of employees of the taxpayer at the end of the taxable year over the number of such employees at the beginning of the taxable year. Defines retained worker to mean any qualified individual who was employed on any date during the taxable year for a period of not less than 52 weeks and whose wages during the last 26 weeks of such period equaled at least 80 of such wages for the first 26 weeks of such period. Defines qualified individual as any individual who: (1) begins employment after 2010 and before 2014, (2) certifies by signed affidavit that such individual has not been employed for 40 hours or more per week during the 60-day period ending on the date such individual begins employment, (3) is not replacing another employee, and (4) is not disqualified for such credit by a relationship to the employer.","title":"To amend the Internal Revenue Code of 1986 to provide a credit to employers for the retention of certain individuals hired before 2013.
"""

test_dataset = SummarizationDataset(
    None, source=[source], source_preprocessing=[tokenize.sent_tokenize],
)
generated_summaries = summarizer.predict(test_dataset, batch_size=1, num_gpus=NUM_GPUS)
print(generated_summaries)

# if os.path.exists(DATA_PATH):
#     shutil.rmtree(DATA_PATH, ignore_errors=True)
# if os.path.exists(CACHE_PATH):
#     shutil.rmtree(CACHE_PATH, ignore_errors=True)