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
        return self.processor(features, sampling_rate=44100,return_tensors='pt').input_values.squeeze()

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
        self.punctuation = set('.,!?\'":;-')

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
        current_word = []
        phones_grouped = []
        # ideally we want to add word boundary symbols to the phones at the start ("_S") and end ("_E") of each word
        # presumably this will require retraining the underlying alignment models (?) which right now I cbf doing
        
        for phone in phones:
            if phone == ' ':
                #if len(current_word) > 1:
                #    current_word[-1] += "_E"
                phones_grouped.append(current_word)
                current_word = []
            else:
                #if len(current_word) == 0:
                #    current_word.append(phone + "_S")
                #else:
                current_word.append(phone)
        if len(current_word) > 0:
            #if len(current_word) > 1:
            #    current_word[-1] += "_E"
            phones_grouped.append(current_word)
        assert len(words) == len(phones_grouped), f"Mismatch between length of words {len(words)} and phones {len(phones_grouped)}"
        return phones_grouped, words
        
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
        # preds is a list of tuples, where each tuple is a phone alignment (start, end, phone)
        # preds [(0.0, 0.15, '[SIL]'), (0.15, 0.24, 'L'), (0.24, 0.33, 'EH'), (0.33, 0.39, 'T'), (0.39, 0.52, 'S'), (0.52, 0.59, 'T'), (0.59, 0.78, 'AA'), (0.78, 0.82, 'R'), (0.82, 0.89, 'T'), (0.89, 0.94, 'W'), (0.94, 1.01, 'IH'), (1.01, 1.04, 'DH'), (1.04, 1.18, 'S'), (1.18, 1.24, 'AH'), (1.24, 1.34, 'M'), (1.34, 1.41, 'B'), (1.41, 1.56, 'EY'), (1.56, 1.66, 'S'), (1.66, 1.71, 'IH'), (1.71, 1.87, 'K'), (1.87, 2.01, 'AW'), (2.01, 2.06, 'N'), (2.06, 2.14, 'T'), (2.14, 2.21, 'IH'), (2.21, 2.39, 'NG'), (2.39, 2.4, '[UNK]'), (2.4, 3.0, '[SIL]')]
        # words ["let's", 'start', 'with', 'some', 'basic', 'counting', '.']
        # phones [['L', 'EH1', 'T', 'S'], ['S', 'T', 'AA1', 'R', 'T'], ['W', 'IH1', 'DH'], ['S', 'AH1', 'M'], ['B', 'EY1', 'S', 'IH0', 'K'], ['K', 'AW1', 'N', 'T', 'IH0', 'NG'], ['.']]

        # IMPORTANT - preds may not contain exactly the same number of elements as the G2P generated phones sequence 
        # this is because silence may have been inserted, or where consecutive words have the same phone at the end/start (e.g. "lets start")

        # words is a list of strings, where each string is a single word in the original sentence (in Chinese, this is either a single character or punctuation)
        # phones is a list of tuples, where each tuple contains the phones for the word at the respective index 
        word_durations = []
        phone_durations = []
        # this is a pointer to the current word/phones tuple
        word_idx = 0
        preds_idx = 0
        preds = [list(x) for x in preds]
        print(preds)
        # iterate over the aligned phones to match them to their respective source words/phones and:
        # 1) replace SIL or UNK with punctuation if applicable, or add the silence to the "word" list
        # 2) calculate the word duration by summing the relevant phone durations
        while preds_idx < len(preds):
            # if a phone was transcribed as "UNK" or "SIL"
            if preds[preds_idx][-1] in ["[UNK]","[SIL]"]:
                # and the word at the respective index is punctuation
                if word_idx < len(words) and words[word_idx] in self.punctuation:
                    word_durations.append([preds[preds_idx][0], preds[preds_idx][1], words[word_idx]])
                    phone_durations.append(word_durations[-1].copy())
                    # if the previous word was punctuation, the start of this phone can sometimes be before the end of the last phone
                    if word_idx > 0:
                        if word_durations[-2][2] in self.punctuation and word_durations[-2][1] > word_durations[-1][0]:
                            start, end = word_durations[-2][0], word_durations[-1][1]
                            dur = (end - start) / 2
                            word_durations[-2][0] = start
                            word_durations[-2][1] = start + dur
                            phone_durations[-2][0] = start 
                            phone_durations[-2][1] = start + dur
                            phone_durations[-1][0] = start + dur
                            phone_durations[-1][1] = end
                            word_durations[-1][0] = start + dur
                            word_durations[-1][1] = end
                    word_idx += 1
                # or the last word was punctuation 
                elif word_idx > 0 and words[word_idx-1] in self.punctuation:
                    # then assign all the silence to the punctuation
                    word_durations[-1][1] = preds[preds_idx][1]
                    phone_durations[-1][1] = preds[preds_idx][1]
                else:
                    print(f"appended {preds[preds_idx]}")
                    word_durations.append(list(preds[preds_idx]))
                    phone_durations.append(word_durations[-1].copy())
                preds_idx += 1
            # otherwise, iterate over every phone in the word at the respective index and add to the list of word phones
            else:
                word_phone_durations = []
                if word_idx >= len(phones):
                    print(phone_durations)
                    print(f"preds_idx is {preds_idx}, pred is {preds[preds_idx]}")
                    raise Exception()

                for word_phone_index, word_phone in enumerate(phones[word_idx]): 
                    if preds[preds_idx][-1] in ["[UNK]", "[SIL]"]:
                        # this is silence in the middle of a word
                        # adjust the start time of  the next prediction and skip
                        if preds_idx < len(preds) - 1:
                            preds[preds_idx+1][0] = preds[preds_idx][0]
                            preds_idx += 1
                        else:
                            break

                    if words[word_idx] in self.punctuation:
                        # means there was punctuation that wasn't aligned properly
                        # allocate half of the last alignment
                        start, end = word_durations[-1][0], word_durations[-1][1]
                        dur = (end - start) / 2
                        word_durations[-1][1] = start + dur
                        phone_durations[-1][1] = word_durations[-1][1]
                        word_phone_durations += [[start + dur, end, words[word_idx]]]
                        break
                    # since there are no word boundaries attached to the aligned phones, we need some way of allocating phone alignments between identical phones at the end/start of consecutive words
                    # here, we adopt a simple solution of assigning half the aligned duration to the end phone of the last word and half to the start phone of the next phone
                    if len(phone_durations) > 0 and word_phone_index == 0 and re.sub("[0-9]", "", word_phone) == re.sub("[0-9]", "", phone_durations[-1][2]):
                        start, end = phone_durations[-1][0], phone_durations[-1][1]
                        duration = (end - start) / 2
                        phone_durations[-1][1] = start + duration
                        word_durations[-1][1] = start + duration
                        word_phone_durations.append([start + duration, end, word_phone])
                    #    print(f"boundary {word_phone_durations[-1]}")
                    else:
                        # if a word has repeated phones, but only one alignment, we allocate equally
                        if len(word_phone_durations) > 0 and word_phone == word_phone_durations[-1][2] and word_phone != preds[preds_idx][2]:
                            start, end = word_phone_durations[-1][0], word_phone_durations[-1][1]
                            dur = (end - start) / 2
                            word_phone_durations[-1][1] = start + dur
                            word_phone_durations.append([start + dur, end, word_phone])
                            #preds_idx += 1
                        else:
                            word_phone_durations.append([preds[preds_idx][0], preds[preds_idx][1], word_phone])
                                                                       
                            print(f"added simple {word_phone_durations[-1]} when pred was {preds[preds_idx]}")
                            if word_phone_index == len(phones[word_idx]) - 1 and word_phone == phones[word_idx+1][0] and  preds[preds_idx+1] != phones[word_idx+1][0]:
                                phone_start, phone_end = word_phone_durations[-1][0], word_phone_durations[-1][1]
                                phone_dur = (phone_end - phone_start) / 2
                                word_phone_durations[-1][1] = phone_start + phone_dur
                                phone_durations += word_phone_durations
                                word_durations.append([word_phone_durations[0][0], phone_start + phone_dur, words[word_idx]])       
                                word_idx += 1 
                                word_phone_durations = [ [ phone_start+phone_dur, phone_end, phones[word_idx][0] ] ]
                                print(f"added repeated {word_phone_durations[-1]} when pred was {preds[preds_idx]}")
                    if preds[preds_idx][-1] == re.sub("[0-9]", "", word_phone):
                        preds_idx += 1
                phone_durations += word_phone_durations
                word_start = word_phone_durations[0][0]
                word_end = word_phone_durations[-1][1]
                word_durations.append([word_start, word_end, words[word_idx]])       
                word_idx += 1
        return word_durations, phone_durations       

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
        self.number_map = {0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九', 10: '十'}
        self.teens_map = {11: '十一', 12: '十二', 13: '十三', 14: '十四', 15: '十五', 16: '十六', 17: '十七', 18: '十八', 19: '十九'}

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
    def num_to_pinyin(self, num_str):
        num = int(num_str)
        if num < 11:
            return self.number_map[num]
        elif 10 < num < 20:
            return self.teens_map[num]
        else:
            thousands = num // 1000
            hundreds = (num % 1000) // 100
            tens = (num % 100) // 10
            ones = num % 10
            result = ''
            if thousands > 0:
                result += self.number_map[thousands] + '千'
                if hundreds == 0 and (tens > 0 or ones > 0):
                    result += '零'
            if hundreds > 0:
                result += self.number_map[hundreds] + '百'
                if tens == 0 and ones > 0:
                    result += '零'
            if tens > 0:
                if tens == 1 and thousands == 0 and hundreds == 0:
                    result += '十'
                else:
                    result += self.number_map[tens] + '十'
            if ones > 0:
                result += self.number_map[ones]
            return result
    
    def convert_numbers_to_pinyin(self, text):
        return re.sub(r'\d+', lambda x: self.num_to_pinyin(x.group()), text) 
   
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
        sen = self.convert_numbers_to_pinyin(sen)
        
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
                        print(f"Ignoring char {char}")
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
        # this is a pointer to the current word/phones tuple
        word_idx = 0
        preds_idx = 0
        # iterate over the aligned phones to match them to their respective source words/phones and:
        # 1) (if applicable) replace SIL or UNK with punctuation
        # 2) otherwise, add SIL or UNK to the list of word/phone alignments
        # 3) calculate the word duration by summing the relevant phone durations
        while preds_idx < len(preds):
            # if a phone was transcribed as silence
            if preds[preds_idx][-1] in ["[UNK]","[SIL]"]:
                # and the current word is punctuation
                if word_idx < len(words) and words[word_idx] in self.punctuation:
                    # then add the punctuation to the list of aligned word/phones and advance to the next word
                    word_durations.append([preds[preds_idx][0], preds[preds_idx][1], words[word_idx]])
                    phone_durations.append(list(word_durations[-1]))
                    word_idx += 1
                # if the previous word was punctuation 
                elif word_idx > 0 and words[word_idx-1] in self.punctuation:
                    # then add the duration of the silence to the phone of the previous
                    word_durations[-1][1] = preds[preds_idx][1]
                    phone_durations[-1] = word_durations[-1]
                # otherwise, just add the silence to the list of aligned words/phones
                else:
                    # if we:
                    # a) encounter silence, and
                    # b) the last phone and the next expected phone are identical, and
                    # c) the next *aligned* phone is different),
                    # then this means the alignment merged consecutive/repeated phones into a single aligned phone 
                    # (e.g. (... (ii4,),(ii4,),(fuu4,)...) produced only (..., (0.5, 0.7, ii4), (0.7,0.8,SIL), (0.8,0.9,fuu4))
                    # we therefore need to perform the same half-half allocation detailed below
                    # we don't need to worry about multi-phone words in Chinese 
                    if preds_idx < len(preds) - 1 and len(phone_durations) > 0:
                        prev_phone = phone_durations[-1]
                        next_phone = phones[word_idx][0]
                        next_aligned = preds[preds_idx+1][-1]
                        is_consecutive_repeated = prev_phone[-1] == next_phone 
                        if is_consecutive_repeated and next_aligned != next_phone:
                            start, end = prev_phone[0], prev_phone[1]
                            duration = (end - start) / 2
                            prev_phone[1] = start + duration
                            word_phone_durations.append([start + duration, end, prev_phone[2]])
                            phone_durations.append(list(word_phone_durations[-1]))
                            word_idx += 1
                    # now add the silence
                    word_durations.append(list(preds[preds_idx]))
                    phone_durations.append(list(word_durations[-1]))
                preds_idx += 1
            # otherwise, iterate over every phone in the word at the respective index and add to the list of word phones
            else:
                word_phone_durations = []
                if word_idx >= len(phones):
                    print(f"phones {phones}")
                    print(f"words {words}")
                    raise Exception()
                elif words[word_idx] in self.punctuation:
                    raise Exception()
                for word_phone in phones[word_idx]: 
                    if preds[preds_idx][-1] in ["[UNK]", "[SIL]"]:
                        raise Exception()

                    # since there are no word boundaries attached to the aligned phones, we need some way of allocating phone alignments between identical phones at the end/start of consecutive words
                    # here, we adopt a simple solution of assigning half the aligned duration to the end phone of the last word and half to the start phone of the next phone
                    if len(phone_durations) > 0 and word_phone == phone_durations[-1][-1]:
                        prev_phone = phone_durations[-1]
                        start, end = prev_phone[0], preds[preds_idx][1]
                        duration = (end - start) / 2
                        prev_phone[1] = start + duration
                        word_phone_durations.append([start + duration, end, word_phone])
                    else:
                        word_phone_durations.append([preds[preds_idx][0], preds[preds_idx][1], word_phone])
                    subbed = re.sub("[0-9]", "", word_phone)
                    
                    if preds[preds_idx][-1] == word_phone:
                        preds_idx += 1

                phone_durations += list(word_phone_durations)
                word_start = word_phone_durations[0][0]
                word_end = word_phone_durations[-1][1]
                word_durations.append([word_start, word_end, words[word_idx]])       
                word_idx += 1
        # when we encounter punctuation followed by [SIL], we just fold the length of the silence into the punctuation phone
        return word_durations, phone_durations       

#    def align_words(self, preds, phones, words):
#
#        word_durations = []
#        phone_durations = []
#        # this is a pointer to the current position within preds
#        i = 0
#        # we need to iterate over the words/phones to match to their respective phone alignments and:
#        # 1) replace SIL or UNK with punctuation if applicable, or add the silence to the "word" list
#        # 2) calculate the word duration by summing the relevant phone durations
#        for word, word_phones in zip(words, phones):
#            while True:
#                align_start, align_end, transcribed_phone = preds[i]
#            
#                # if transcribed as "UNK" or "SIL"
#                if transcribed_phone == "[UNK]" or transcribed_phone == "[SIL]":
#                    # this cannot be in the middle of a word, so if our stack is non-empty, add it to the list 
#                    # this will always be treated as a "word", irrespective of whether it was punctuation or not
#                    word_durations.append(preds[i])
#                    i += 1
#                    # and if our current word was punctuation
#                    if word in self.punctuation:
#                        phone_durations.append((word_durations[-1][0], word_durations[-1][1], word))
#                        break
#                # otherwise
#                else:
#                    word_phone_durations = []
#                    for phone in word_phones:
#                        word_phone_durations.append((preds[i][0], preds[i][1], phone))
#                        i += 1
#                    phone_durations += word_phone_durations
#                    word_start = word_phone_durations[0][0]
#                    word_end = word_phone_durations[-1][1]
#                    word_durations.append((word_start, word_end, word))        
#                    break
#        print(f"word_durations {word_durations} phone_durations {phone_durations} i {i} len preds {len(preds)} preds {preds}")
#        # this just captures any trailing silence at the end of the sentence
#        if i < len(preds):
#            assert i == len(preds) - 1
#            word_durations.append(preds[-1])
#        elif i != len(preds):
#            raise Exception()
#        # when we encounter punctuation followed by [SIL], we just fold the length of the silence into the punctuation phone
#        return word_durations, phone_durations       
        


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
    
    

    
