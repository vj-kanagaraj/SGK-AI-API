import mammoth
from bs4 import BeautifulSoup
from textblob import TextBlob
from .excel_processing import *

class msd_extraction(base):
    def __init__(self):
        super().__init__()
        self.file_name = None
        self.regex_heading_msd = r"^\d+\.\d?[\-\s][^%ml]|\<li\>"
        self.final = None
        self.validation_categories = {
                                      'warning':['warning'],
                                      'storage_instructions':['storage_instructions'],
                                      'manufacturer':['address','manufacturer'],
                                      'marketing_company': ['address','marketing_company'],
                                      'expiry_date':['expiry_date'],
                                      'form_content':['form_content'],
                                      'method_route':['method_route'],
                                      'others': ['others'],
                                      }

    def docx_to_html(self,file,method=None):
        print('entering docx to html')
        if file.startswith('\\'):
            print('connecting to SMB share')
            try:
                with smbclient.open_file(r"{}".format(file), mode='rb', username=smb_username,
                                         password=smb_password) as f:
                    html = mammoth.convert_to_html(f).value
                    print('file found')
            except:
                smbclient.reset_connection_cache()
                with smbclient.open_file(r"{}".format(file), mode='rb', username=smb_username,
                                         password=smb_password) as f:
                    html = mammoth.convert_to_html(f).value
                    print('file found')
            finally:
                smbclient.reset_connection_cache()
        else:
            print('local')
            file = document_location + file
            html = mammoth.convert_to_html(file).value
        return html

    def text_from_html(self,file_name):             # only for MSD
        print('entering text to _html')
        html = self.docx_to_html(file_name)
        # print('html---->',html)
        soup = BeautifulSoup(html, 'html.parser')
        '''paragraphs = soup.find_all(['p', 'li'])
        for para in paragraphs:
            if '<em>' not in str(para):
                yield para'''
        if ('<table>' in html) and ('<em>' in html) and all([len(_row.find_all('td')) != 1 for _row in soup.find_all('tr')]):                         # for table structure
            print('inside table structure')
            rows = soup.find_all('tr')
            for row in rows:
                for column in row.find_all('td'):
                    # print('column----->',column)
                    # yield column.findall('p')
                    # print(f'column---------->{column}')
                    yield column
        else:                                       # for heading - value structure
            print('inside normal structure')
            paragraphs = soup.find_all(['p', 'li'])
            for para in paragraphs:
                # print('para------>',para)
                yield para

    def heading_value_extraction2(self,file_name):
        tmp = []
        key = ''
        generator = self.text_from_html(file_name)
        paragraphs = [component for component in generator]
        for i, content in enumerate(paragraphs):
            text = str(content).strip()
            if '' in tmp:
                tmp.remove('')
            if re.findall(self.regex_heading_msd, content.text.strip()) or '<li>' in text:
                if key and tmp:
                    yield key , ["$$".join(tmp)]
                key = content.text.strip()
                # print(f'key----->{key}')
                tmp.clear()
            else:
                if i == len(paragraphs) - 1:
                    if text and not re.findall(r"Panel\s\d", text):
                        try:
                            content = BeautifulSoup(str(content), 'html.parser')
                            for para in content.find_all('p'):
                                # print('para----yyyyyyyyy', para)
                                para = str(para)
                                para = para.replace('<strong>', '<b>').replace('</strong>', '</b>')
                                para = re.sub(r"<(\/?[^/bems]).*?>", '', para)
                                tmp.append(para)
                        except:
                            pass
                    if key and tmp:
                        yield key, ['$$'.join(tmp)]
                        tmp.clear()
                else:
                    if text and not re.findall(r"Panel\s\d", text):  # filter out heading like 'big panel 1'
                        try:
                            content = BeautifulSoup(str(content),'html.parser')
                            for para in content.find_all('p'):
                                para = str(para)
                                # print('para----xxxxxxx',para)
                                para = para.replace('<strong>', '<b>').replace('</strong>', '</b>')
                                para = re.sub(r"<(\/?[^/bems]).*?>", '', para)
                                tmp.append(para)
                        except:
                            pass

    def validation(self):
        print('inside validation')
        for category , cate_value in self.validation_categories.items():
            if category in self.final:
                for index,value in enumerate(self.final[category]):
                    for lang_key, val in value.items():
                        # output = base('msd_content',msd_content_model_location).prediction(val,method='labse')
                        output = base('msd_content',msd_content_model_location).prediction(val)
                        pred = output['output']
                        probability = output['probability']
                        # print(val,'------>',pred)
                        if pred in cate_value:
                            pass
                        else:
                            try:
                                undetected_msd_log(file_name=self.file_name,text=val,header_category=category,content_category=pred,language_code=lang_key).save()
                            except:
                                pass
                            if pred == 'None':
                                pass
                            elif pred != 'None' and probability > 0.70:
                                self.final[category].pop(index)
                                if pred in self.final:
                                    self.final[pred].append({lang_key:val})
                                else:
                                    self.final[pred] = [{lang_key:val}]
                            else:
                                pass
                                # final[category].pop(index)
                            print('fail')
                    if not self.final[category]:
                        self.final.pop(category)

    def main(self,file_name):
        final = {}
        all_lang = set()
        for key, value in self.heading_value_extraction2(file_name):
            # prediction = base('msd', msd_model_location).prediction(key,method='labse')['output']
            prediction = base('msd', msd_model_location).prediction(key)['output']
            if prediction == 'None':
                try:
                    blob = TextBlob(key)
                    key = blob.translate(to='en')
                    prediction = base('msd', msd_model_location).prediction(str(key))['output']
                except:
                    pass
            if prediction in msd_categories_lang_exception:
                val = value[0].replace('$$', '\n')
                # val = ' '.join([f"<p>{_val.replace('$$','')}</p>" for _val in value[0].split('$$')])
                # cleaned_text = val.translate(str.maketrans("","",string.punctuation))
                # cleaned_text = re.sub(r'\d','',cleaned_text).lower()
                cleaned_text = re.sub(r'\d','',val).lower()
                cleaned_text = cleaned_text.replace('\n', ' ').replace(':', '').strip()
                print(cleaned_text)
                try:
                    lang = lang_detect(cleaned_text)
                    print(f'lang_detect------>{cleaned_text} ------>{lang}')
                    # lang_blob = TextBlob(cleaned_text)
                    # lang1 = lang_blob.detect_language()
                    # print(f'textblob---->{cleaned_text} ------>{lang1}')
                except:
                    lang = classify(cleaned_text)[0]
                    print(f'classify---->{cleaned_text} ------>{lang}')
                all_lang.add(lang)
                if prediction in final:
                    final[prediction].append({lang: str(val)})
                else:
                    final[prediction] = [{lang: str(val)}]
            else:
                for para in value[0].split('$$'):
                    # cleaned_text = para.translate(str.maketrans("", "", string.punctuation))
                    # cleaned_text = re.sub(r'\d', '', cleaned_text).lower()
                    cleaned_text = re.sub(r'\d', '', para).lower()
                    cleaned_text = cleaned_text.replace('\n',' ').replace(':','').strip()
                    try:
                        lang = lang_detect(cleaned_text)
                        print(f'lang_detect------>{cleaned_text} ------>{lang}')
                        # lang_blob = TextBlob(cleaned_text)
                        # lang1 = lang_blob.detect_language()
                        # print(f'textblob---->{cleaned_text} ------>{lang1}')
                    except:
                        lang = classify(cleaned_text)[0]
                        print(f'classify---->{cleaned_text} ------>{lang}')
                    all_lang.add(lang)
                    if prediction in final:
                        final[prediction].append({lang: para})
                    else:
                        final[prediction] = [{lang: para}]
        # print('final---->',final)
        if 'None' in final:
            final['unmapped'] = final['None']
            final.pop('None', None)
        # self.final = {**{'status': 1, 'language': list(all_lang), 'file_name': [file_name]}, **final}
        self.final = final
        # print('before validation',self.final)
        self.validation()
        self.final = {**{'status': 1, 'language': list(all_lang), 'file_name': [file_name]}, **self.final}
        # print('after validation',self.final)
        return self.final

