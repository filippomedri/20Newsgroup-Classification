import re
import logging
import pandas as pd
import numpy as np
import os
import ntpath
from gensim.utils import simple_preprocess
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from newsgroup_conf import Config

class ETL(object):

    def __init__(self):
        self.config = Config()
        self.header_tag = \
            [ r'Path:.*\n',
              r'From:.*\n',
              r'Sender:.*\n',
              r'Reply-To:.*\n',
              r'Newsgroups:.*\n',
              r'Subject:.*\n',
              r'Date:.*\n',
              r'Organization:.*\n',
              r'Lines:.*\n',
              r'Keywords:.*\n',
              r'X-Newsreader:.*\n',
              r'Message-ID:.*\n',
              r'Article-I.D.:.*\n',
              r'References:.*\n',
              r'NNTP-Posting-Host:.*\n',
              r'Xref:.*\n',
              r'Followup-To:.*\n',
              r'Distribution:.*\n',
              r'Originator:.*\n',
              r'Nntp-Posting-Host:.*\n',
              r'Internet:.*\n',
              r'Bitnet:.*\n',
              r'In-reply-to:.*\n',
              r'UUCP:.*\n',
              r'X-Mailer:.*\n',
              r'Email :.*\n',
              r'e-mail .*\n'
              r'e-mail .*\n',
              r'Summary:.*\n',
              r'^.* wrote:\n',
              r'^.* writes:\n',
              r'^.* says:\n',
              r'^.* sad:\n',
              r'^.* All:\n',
              r'^.* all:\n',
              r'[^@]+@[^@]+\.[^@]+',
              r'^(?!:\/\/)([a-zA-Z0-9-_]+\.)*[a-zA-Z0-9][a-zA-Z0-9-_]+\.[a-zA-Z]{2,11}?$'
              ]

        self._QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


    def __strip_header_quotes_and_footer(self, document_lines):
        output_agg = list()

        # First eliminate quotation
        good_lines = [line for line in document_lines
                      if not self._QUOTE_RE.search(line)]

        # Then Header and Footer Patterns
        for line in good_lines:
            for tag in self.header_tag:
                line = re.sub(tag,'',line)

            if (line != ''):
                print(line)
                output_agg.append(line)
        document_string = '\n'.join(output_agg)
        return document_string

    def __find_ngrams(self,input_list, n):
        # Courtesy http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
        ngrams = list(zip(*[input_list[i:] for i in range(n)]))
        flattened_ngrams = list(map(lambda x: '_'.join(x), ngrams))
        return flattened_ngrams

    def __eliminate_stop_words_and_plurals(self, input_list):
        print(input_list)
        print()
        p = nltk.PorterStemmer()
        stop_words.add('like')
        return [p.stem(token) for token in input_list if not token in stop_words]

    def extract(self):
        """
        Extract necessary data / resources from upstream. This method will:
         - Validate that newsgroup data set is available, and read in
        """
        logging.info('Begin extract')

        # Extract resources from file system

        # Newsgroup20: Get list of all candidate documents
        newsgroup_path = self.config.get_property('newsgroup_path')
        logging.info('Loading file from: {}'.format(newsgroup_path))

        document_paths = [os.path.join(d, x)
            for d, dirs, files in os.walk(newsgroup_path)
            for x in files]

        # Newsgroup20: Create observations data set
        self.observations = pd.DataFrame(document_paths, columns=['document_path'])
        logging.info('Shape of observations data frame created from glob matches: {}'.format(self.observations.shape))

        # Newsgroup20: Re-order rows
        observations = self.observations.sample(frac=1)

        # Newsgroup20: Subset number of observations, if it's a test run
        if self.config.get_property('test_run'):
            logging.info('Reducing file size for test run')
            self.observations = observations.sample(100)
            self.observations = observations.reset_index()
            logging.info('Test run number of records: {}'.format(len(observations.index)))

        logging.info('End extract')


    def transform(self):
        logging.info('Begin transform')
    
        # Transform newsgroup20 data set
        # Newsgroup20: Extract article filename from document path
        self.observations['filename'] = self.observations['document_path'].apply(lambda x: ntpath.basename(x))
    
        # Newsgroup20: Extract article category from document path
        self.observations['category'] = self.observations['document_path'].apply(lambda x: ntpath.basename(os.path.dirname(x)))

        # Newsgroup20: Extract article text (and strip article headers), from document path

        self.observations['text'] = self.observations['document_path'].apply(lambda x: self.__strip_header_quotes_and_footer(list(open(x, encoding="latin-1"))))

        # Remove non-ascii characters
        #observations['text'] = observations['text'].apply(lambda x: x.decode('ascii', errors='ignore'))
        self.observations['text'] = self.observations['text'].apply(lambda x: re.sub(r'[^\x00-\x7f]',r'',x))
    
        # Newsgroup20: Convert text to normalized tokens. Unknown tokens will map to 'UNK'.
        self.observations['tokens'] = self.observations['text'].apply(simple_preprocess)

        # Newsgroup20: Eliminate Stop Word
        self.observations['tokens'] = self.observations['tokens'].apply(lambda x:self.__eliminate_stop_words_and_plurals(x))

        # Newsgroup20: Create bigrams
        self.observations['bigrams'] = self.observations['tokens'].apply(lambda x: self.__find_ngrams(x, n=2))

        # Newsgroup20: Create modeling text
        #self.observations['modeling_text_list'] = self.observations['tokens'] + self.observations['bigrams']
        self.observations['modeling_text_list'] = self.observations['tokens']
        self.observations['modeling_text'] = self.observations['modeling_text_list'].apply(lambda x: ' '.join(x))

        print(self.observations.head())

        '''
        lib.archive_dataset_schemas('transform', locals(), globals())
        logging.info('End transform')
        return observations
        '''