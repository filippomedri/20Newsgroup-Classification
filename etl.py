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
        ## read configuration
        self.config = Config()
        # tag to remove from the e-mail text
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
        # regular expression to remove the introduction of the quote
        self._QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')

    def __strip_header_quotes_and_footer(self, document_lines):
        '''
        :param document_lines:
        :return: text of e-mail without header, footer and quotes
        '''
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
        '''
        :param input_list: list of token words
        :return: the list of token without english stop words
        words in plural are converted to singular
        '''
        p = nltk.PorterStemmer()
        return [p.stem(token) for token in input_list if not token in stop_words]

    def extract(self):
        """
        Read the newsgroup dataset from the filesystem
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

        logging.info('End extract')

    def transform(self):
        '''
        Perform transformation on the 20 Newsgroup Dataset:
        Create the following columns of the observations dataframe
        - filename              : the name of the file
        - category              : the name of the newsgroup that file belongs to
        - text                  : the text of the file
        - tokens                : the tokens which made up the file
        - modeling text list    : the text used by the modeling algorithms
        '''
        logging.info('Begin transform')
    
        # Transform newsgroup20 data set
        # Newsgroup20: Extract article filename from document path
        self.observations['filename'] = self.observations['document_path'].apply(lambda x: ntpath.basename(x))
    
        # Newsgroup20: Extract article category from document path
        self.observations['category'] = self.observations['document_path'].apply(lambda x: ntpath.basename(os.path.dirname(x)))

        # Newsgroup20: Extract article text (and strip article headers), from document path
        self.observations['text'] = self.observations['document_path'].apply(lambda x: self.__strip_header_quotes_and_footer(list(open(x, encoding="latin-1"))))

        # Remove non-ascii characters
        self.observations['text'] = self.observations['text'].apply(lambda x: re.sub(r'[^\x00-\x7f]',r'',x))
    
        # Newsgroup20: Convert text to normalized tokens. Unknown tokens will map to 'UNK'.
        self.observations['tokens'] = self.observations['text'].apply(simple_preprocess)

        # Newsgroup20: Eliminate Stop Word
        self.observations['tokens'] = self.observations['tokens'].apply(lambda x:self.__eliminate_stop_words_and_plurals(x))

        # Newsgroup20: Create modeling text
        self.observations['modeling_text'] = self.observations['tokens'].apply(lambda x: ' '.join(x))

        logging.info('End transform')

    def load(self):
        '''
        load clustering data and modeling list data to 2 different csv file
        '''
        self.observations[['component_1', 'component_2', 'component_3', 'category', 'cluster']].to_csv('clustering.csv')
        self.observations[['modeling_text_list']].to_csv('text.csv')