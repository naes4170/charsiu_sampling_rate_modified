#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import numpy as np
from itertools import groupby, chain
import soundfile as sf
import librosa.core
import unicodedata
from builtins import str as unicode
from nltk.tokenize import TweetTokenizer
word_tokenize = TweetTokenizer().tokenize

from g2p_en import G2p
from g2p_en.expand import normalize_numbers 
from g2pM import G2pM
from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor, Wav2Vec2Processor



class CharsiuPreprocessor:
    
    def __init__(self):
        pass
    
    
    def get_phones_and_words(self):
        raise NotImplementedError


    def get_phone_ids(self):
        raise NotImplementedError
        
        
    def mapping_phone2id(self,phone):
        '''
        Convert a phone to a numerical id

        Parameters
        ----------
        phone : str
            A phonetic symbol

        Returns
        -------
        int
            A one-hot id for the input phone

        '''
        return self.processor.tokenizer.convert_tokens_to_ids(phone)
    
    def mapping_id2phone(self,idx):
        '''
        Convert a numerical id to a phone

        Parameters
        ----------
        idx : int
            A one-hot id for a phone

        Returns
        -------
        str
            A phonetic symbol

        '''
        
        return self.processor.tokenizer.convert_ids_to_tokens(idx)
        
    
    def audio_preprocess(self,audio,sr=16000):
        '''
        Load and normalize audio
        If the sampling rate is incompatible with models, the input audio will be resampled.

        Parameters
        ----------
        path : str
            The path to the audio
        sr : int, optional
            Audio sampling rate, either 16000 or 32000. The default is 16000.

        Returns
        -------
        torch.Tensor [(n,)]
            A list of audio sample as an one dimensional torch tensor

        '''
        if type(audio)==str:
            features, sr = librosa.core.load(audio,sr=16000)
            assert sr == 16000
            
        elif isinstance(audio, np.ndarray):
            features = audio
        else:
            raise Exception('The input must be a path or a numpy array!')
        return self.processor(features, sampling_rate=16000,return_tensors='pt').input_values.squeeze()

'''
English g2p processor
'''
class CharsiuPreprocessor_en(CharsiuPreprocessor):
    
    def __init__(self):
        
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('charsiu/tokenizer_en_cmu')
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.g2p = G2p()
        self.sil = '[SIL]'
        self.sil_idx = self.mapping_phone2id(self.sil)
        self.punctuation = set('.,!?')

    def get_phones_and_words(self,sen):
        '''
        Convert texts to phone sequence

        Parameters
        ----------
        sen : str
            A str of input sentence

        Returns
        -------
        sen_clean : list
            A list of phone sequence without stress marks
        sen : list
             A list of phone sequence with stress marks
             
    
        xxxxx should sen_clean be deleted?

        '''     
        
        phones = self.g2p(sen)
        words = self._get_words(sen)
        phones = list(tuple(g) for k,g in groupby(phones, key=lambda x: x != ' ') if k)  
        
        aligned_phones = []
        aligned_words = []
        for p,w in zip(phones,words):
            if re.search(r'\w+\d?',p[0]):
                aligned_phones.append(p)
                aligned_words.append(w)
            elif p[0] in self.punctuation:
                aligned_phones.append((p[0],))
                aligned_words.append((p[0]))
#        print("aligned_words {aligned_words
        assert len(aligned_words) == len(aligned_phones)
        return aligned_phones, aligned_words
        
    def get_phone_ids(self,phones,append_silence=True):
        '''
        Convert phone sequence to ids

        Parameters
        ----------
        phones : list
            A list of phone sequence
        append_silence : bool, optional
            Whether silence is appended at the beginning and the end of the sequence. 
            The default is True.

        Returns
        -------
        ids: list
            A list of one-hot representations of phones

        '''
        phones = list(chain.from_iterable(phones))
        ids = [self.mapping_phone2id(re.sub(r'\d','',p)) for p in phones]

        # append silence at the beginning and the end
        if append_silence:
            if ids[0]!=self.sil_idx:
                ids = [self.sil_idx]+ids
            if ids[-1]!=self.sil_idx:
                ids.append(self.sil_idx)
        return ids 
    
    
    
    def _get_words(self,text):
        '''
        from G2P_en
        https://github.com/Kyubyong/g2p/blob/master/g2p_en/g2p.py

        Parameters
        ----------
        sen : TYPE
            DESCRIPTION.

        Returns
        -------
        words : TYPE
            DESCRIPTION.

        '''
        
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        words = word_tokenize(text)
        
        return words
    
    def align_words(self, preds, phones, words):
    
        words_rep = [w for ph,w in zip(phones,words) for p in ph]
        phones_rep = [re.sub(r'\d','',p) for ph,w in zip(phones,words) for p in ph]
        assert len(words_rep)==len(phones_rep)

        # match each phone to its word
        word_dur = []
        count = 0

        phones_with_punctuation = []

        for dur in preds:
            if dur[-1] == '[SIL]':
                if len(phones_with_punctuation) > 0 and phones_with_punctuation[-1][-1] in self.punctuation:
                    print(f"SIL after punc {phones_with_punctuation[-1][-1]}")
                    phones_with_punctuation[-1][1] = dur[1]
                else:
                    word_dur.append((dur,'[SIL]'))
                    phones_with_punctuation.append(dur)
            elif dur[-1] == "[UNK]":
                if phones_rep[count+1] in self.punctuation:
                    dur = (dur[0], dur[1], phones_rep[count+1])
                    word_dur.append((dur,phones_rep[count+1]))
                    phones_with_punctuation.append(dur)
                    count += 1
                else:
                    raise Exception(f"{phones_rep[count+1]} not punctuation")
            else:
                while dur[-1] != phones_rep[count]:
                    count += 1
    
                phones_with_punctuation.append(dur)
                word_dur.append((dur,words_rep[count])) #((start,end,phone),word)
        # merge phone-to-word alignment to derive word duration
        words = []
        for key, group in groupby(word_dur, lambda x: x[-1]):
            group = list(group)
            entry = (group[0][0][0],group[-1][0][1],key)
            words.append(entry)
           
        return words, phones_with_punctuation


'''
Mandarin g2p processor
'''


class CharsiuPreprocessor_zh(CharsiuPreprocessor_en):

    def __init__(self):
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('charsiu/tokenizer_zh_pinyin')
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.g2p = G2pM()
        self.sil = "[SIL]"
        self.sil_idx = self.mapping_phone2id(self.sil)
        self.punctuation = set('.,!?。，！？、"”“。\'《》`')
        # Pinyin tables
        self.consonant_list = set(['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k',
                  'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z',
                  'c', 's'])

        self.transform_dict = {'ju':'jv', 'qu':'qv', 'xu':'xv','jue':'jve',
                          'que':'qve', 'xue':'xve','quan':'qvan',
                          'xuan':'xvan','juan':'jvan',
                          'qun':'qvn','xun':'xvn', 'jun':'jvn',
                             'yuan':'van', 'yue':'ve', 'yun':'vn',
                            'you':'iou', 'yan':'ian', 'yin':'in',
                            'wa':'ua', 'wo':'uo', 'wai':'uai',
                            'weng':'ueng', 'wang':'uang','wu':'u',
                            'yu':'v','yi':'i','yo':'io','ya':'ia', 'ye':'ie', 
                            'yao':'iao','yang':'iang', 'ying':'ing', 'yong':'iong',
                            'yvan':'van', 'yve':'ve', 'yvn':'vn',
                            'wa':'ua', 'wo':'uo', 'wai':'uai',
                            'wei':'ui', 'wan':'uan', 'wen':'un', 
                            'weng':'ueng', 'wang':'uang','yv':'v',
                            'wuen':'un','wuo':'uo','wuang':'uang',
                            'wuan':'uan','wua':'ua','wuai':'uai',
                            'zhi':'zhiii','chi':'chiii','shi':'shiii',
                            'zi':'zii','ci':'cii','si':'sii'}
        self.er_mapping ={'er1':('e1','rr'),'er2':('e2','rr'),'er3':('e3','rr'),'er4':('e4','rr'),
                          'er5':('e5','rr'),'r5':('e5','rr')}
        self.rhyme_mapping = {'iu1':'iou1','iu2':'iou2','iu3':'iou3','iu4':'iou4','iu5':'iou5',
                              'u:e1':'ve1','u:e2':'ve2','u:e3':'ve3','u:e4':'ve4','u:e5':'ve5',
                              'u:1':'v1','u:2':'v2','u:3':'v3','u:4':'v4','u:5':'v5',
                              'ueng1':('u1','eng1'),'ueng2':('u2','eng2'),'ueng3':('u3','eng3'),
                              'ueng4':('u4','eng4'),'ueng5':('u5','eng5'),'io5':('i5','o5'),
                              'io4':('i4','o4'),'io1':('i1','o1')}
        
    def get_phones_and_words(self,sen):
        '''
        Convert texts to phone sequence

        Parameters
        ----------
        sen : str
            A str of input sentence

        Returns
        -------
        sen_clean : list
            A list of phone sequence without stress marks
        sen : list
             A list of phone sequence with stress marks
        
        xxxxx should sen_clean be removed?
        '''     
        # sen is a plain string of Chinese (possibly containing punctuation, including whitespace)
        # first, strip the whitespace because we don't need it
        sen = sen.strip().replace(" ", "") 
        # then, call G2P to phones (numeric pinyin)
        phones = self.g2p(sen.strip())
        aligned_phones = []
        aligned_words = []

        # a single UTF16 character in [sen] will now correspond to a single phone, with the exception of punctuation

        word_idx = 0
        for phone in phones:
            # non-punctuation
            if re.search(r'\w+:?\d',phone):
                aligned_phones.append(self._separate_syllable(self.transform_dict.get(phone[:-1],phone[:-1])+phone[-1]))
                aligned_words.append(sen[word_idx])               
                word_idx += 1           
            else:
                for char in phone:
                    if char in self.punctuation:
                        aligned_phones.append((char,))
                        aligned_words.append((char))
                    else:
                        print(f"Ignoring char {p}")
                    word_idx += 1
        assert len(aligned_phones)==len(aligned_words)
        return aligned_phones, aligned_words


    def get_phone_ids(self,phones,append_silence=True):
        '''
        Convert phone sequence to ids

        Parameters
        ----------
        phones : list
            A list of phone sequence
        append_silence : bool, optional
            Whether silence is appended at the beginning and the end of the sequence. 
            The default is True.

        Returns
        -------
        ids: list
            A list of one-hot representations of phones

        '''
        phones = list(chain.from_iterable(phones))
        ids = [self.mapping_phone2id(p) for p in phones]

        # append silence at the beginning and the end
        if append_silence:
            if ids[0]!=self.sil_idx:
                ids = [self.sil_idx]+ids
            if ids[-1]!=self.sil_idx:
                ids.append(self.sil_idx)
        return ids 
    
    
    def _separate_syllable(self,syllable):
        """
        seprate syllable to consonant + ' ' + vowel

        Parameters
        ----------
        syllable : xxxxx TYPE
            xxxxx DESCRIPTION.

        Returns
        -------
        syllable: xxxxx TYPE
            xxxxxx DESCRIPTION.

        """
        
        assert syllable[-1].isdigit(), f"Expected numeric tone for {syllable}"
        if syllable == 'ri4':
            return ('r','iii4')
        if syllable[:-1] == 'ueng' or syllable[:-1] == 'io':
            return self.rhyme_mapping.get(syllable,syllable)
        if syllable in self.er_mapping.keys():
            return self.er_mapping[syllable]
        if syllable[0:2] in self.consonant_list:
            #return syllable[0:2].encode('utf-8'),syllable[2:].encode('utf-8')
            return syllable[0:2], self.rhyme_mapping.get(syllable[2:],syllable[2:])
        elif syllable[0] in self.consonant_list:
            #return syllable[0].encode('utf-8'),syllable[1:].encode('utf-8')
            return syllable[0], self.rhyme_mapping.get(syllable[1:],syllable[1:])
        else:
            #return (syllable.encode('utf-8'),)
            return (syllable,)
 
    def align_words(self, preds, phones, words):
        # preds is a list of tuples, where each tuple is a phone alignment (start, end, phone)
        # > preds
        # [(0.0, 0.1, '[SIL]'), (0.1, 0.26, 'q'), (0.26, 0.37, 'ing3'), (0.37, 0.57, 'sh'), (0.57, 0.99, 'uo1'), (0.99, 1.0, '[UNK]'), (1.0, 1.13, '[SIL]'), (1.13, 1.24, 'n'), (1.24, 1.25, 'i3'), (1.25, 1.41, 'h'), (1.41, 1.62, 'ao3'), (1.62, 1.64, '[UNK]'), (1.64, 2.45, '[SIL]')]

        # words is a list of strings, where each string is a single word in the original sentence (in Chinese, this is either a single character or punctuation)
        # >> words 
        #  ['请', '说', '“', '你', '好', '”', '。'] 
        # phones is a list of tuples, where each tuple contains the phones for the word at the respective index 
        # >> phones 
        # [('q', 'ing3'), ('sh', 'uo1'), ('“',), ('n', 'i3'), ('h', 'ao3'), ('”',), ('。',)] 

        word_durations = []
        phone_durations = []
        # this is a pointer to the current position within preds
        i = 0
        # we need to iterate over the words/phones to match to their respective phone alignments and:
        # 1) replace SIL or UNK with punctuation if applicable, or add the silence to the "word" list
        # 2) calculate the word duration by summing the relevant phone durations
        for word, word_phones in zip(words, phones):
            while True:
                align_start, align_end, transcribed_phone = preds[i]
            
                # if transcribed as "UNK" or "SIL"
                if transcribed_phone == "[UNK]" or transcribed_phone == "[SIL]":
                    # this cannot be in the middle of a word, so if our stack is non-empty, add it to the list 
                    # this will always be treated as a "word", irrespective of whether it was punctuation or not
                    word_durations.append(preds[i])
                    i += 1
                    # and if our current word was punctuation
                    if word in self.punctuation:
                        phone_durations.append((word_durations[-1][0], word_durations[-1][1], word))
                        break
                # otherwise
                else:
                    word_phone_durations = []
                    for phone in word_phones:
                        word_phone_durations.append((preds[i][0], preds[i][1], phone))
                        i += 1
                    phone_durations += word_phone_durations
                    word_start = word_phone_durations[0][0]
                    word_end = word_phone_durations[-1][1]
                    word_durations.append((word_start, word_end, word))        
                    break
        print(f"word_durations {word_durations} phone_durations {phone_durations}")
        assert i == len(preds)
        # when we encounter punctuation followed by [SIL], we just fold the length of the silence into the punctuation phone
        return word_durations, phone_durations       
        


if __name__ == '__main__':
    '''
    Testing functions
    '''    
    
    processor = CharsiuPreprocessor_zh()
    phones, words = processor.get_phones_and_words("鱼香肉丝、王道椒香鸡腿和川蜀鸡翅。")    
    print(phones)
    print(words)
    ids = processor.get_phone_ids(phones)
    print(ids)

    processor = CharsiuPreprocessor_en()
    phones, words = processor.get_phones_and_words("I’m playing octopath right now!")    
    print(phones)
    print(words)
    ids = processor.get_phone_ids(phones)
    print(ids)
    
    

    
