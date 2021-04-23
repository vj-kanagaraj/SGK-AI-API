from __future__ import unicode_literals
import io
import pdfplumber
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_text_to_fp
from sklearn.metrics.pairwise import cosine_similarity
import pdfminer
from bs4 import BeautifulSoup
from textblob import TextBlob
import tempfile
import pikepdf
from pdf2docx import parse
import mammoth
from bidi.algorithm import get_display


from .excel_processing import *

# header_dict_value = {header: laser.embed_sentences(content,lang='en').mean(0).reshape(1,1024) for header, content in header_dict.items()}

class ferrero_extraction(base):
    def __init__(self):
        super().__init__()
        self.input_pdf_holder = None
        self.nutrition_table_title = ['Nutrition Information', 'nutrition declaration','Part D1 (LTR) - Nutrition Information','nutrition information typical values']
        self.output_io = io.StringIO()
        self.input_file = io.BytesIO()
        self.temp_directory = tempfile.TemporaryDirectory(dir=document_location)
        self.input_pdf_location = f'{self.temp_directory.name}/input_pdf.pdf'

    def get_pdf_file(self,input_pdf):
        if input_pdf.startswith('\\'):
            print('connecting to SMB share')
            try:
                with smbclient.open_file(r"{}".format(input_pdf), mode='rb', username=smb_username,
                                         password=smb_password) as f:
                    with open(self.input_pdf_location,mode='wb') as op:
                        op.write(f.read())
                    print('file found')
            except:
                smbclient.reset_connection_cache()
                with smbclient.open_file(r"{}".format(input_pdf), mode='rb', username=smb_username,
                                         password=smb_password) as f:
                    with open(self.input_pdf_location,mode='wb') as op:
                        op.write(f.read())
                    print('file found')

            finally:
                # pike_input_pdf = pikepdf.open(self.input_file)
                # pike_input_pdf.save(self.input_pdf_location)
                self.input_pdf_holder = self.input_pdf_location
            return self.input_pdf_holder
        else:
            self.input_pdf_holder = "".join((document_location,input_pdf))
            return self.input_pdf_holder

    def pdf_to_docx_to_html(self,input_pdf, page_no):  # 3
        docx_name = f'{self.temp_directory.name}/{page_no}.docx'
        pdf_obj = pdfplumber.open(input_pdf)
        if not os.path.exists(docx_name) and page_no-1 in range(len(pdf_obj.pages)):
            parse(input_pdf, docx_name, start=page_no - 1, end=page_no)
            x = mammoth.convert_to_html(docx_name, style_map="b => b").value
            html = BeautifulSoup(x, 'html.parser')
        else:
            x = ""
            html = BeautifulSoup(x, 'html.parser')
        return html

    def is_nutrition_table(self,text):  # 4
        if str(text).strip():
            similarity = cosine_similarity(laser.embed_sentences(text, lang='en'),
                                           laser.embed_sentences(self.nutrition_table_title, lang='en').mean(0).reshape(1,1024))[0][0]
        else:
            return False
        if similarity > 0.80:
            return True
        else:
            pass

    def header_similarity(self,text):
        embed_text = laser.embed_sentences([text], lang='en')
        similarity = lambda t: {key: cosine_similarity(value, t)[0][0] for key, value in header_dict_value.items()}
        score_dict = similarity(embed_text)
        max_cate = max(score_dict, key=score_dict.get)
        if score_dict and score_dict[max(score_dict, key=score_dict.get)] > 0.80:
            return max_cate
        else:
            return None

    '''def pdf_to_tables(self,input_pdf, page_no):  # pdfplumber 2
        if input_pdf.startswith('\\'):
            if not self.input_pdf_holder:
                if self.input_file.getvalue():
                    pdf = pdfplumber.open(self.input_file)
                    self.input_pdf_holder = pdf
                else:
                    pdf = pdfplumber.open(self.get_pdf_file(input_pdf))
            else:
                print('inside self.pdf')
                pdf = self.input_pdf_holder
        else:
            pdf = pdfplumber.open(input_pdf)
        page = pdf.pages[page_no - 1]
        tables = page.extract_tables()
        for table_no in range(len(tables)):
            # print(tables)
            # print('------'*10)
            yield tables[table_no]'''

    def pdf_to_docx_to_tables(self,input_pdf, page_no):  # pdfplumber 2
        html = self.pdf_to_docx_to_html(input_pdf,int(page_no))
        for table in html.find_all('table'):
            _table = []
            for row in table.find_all('tr'):
                _row = []
                for column in row.find_all('td'):
                    col = re.sub(r'<(?!b|\/b).*?>','',str(column))
                    col = re.sub(r'\<\/?br\/?\>','\n',str(col))
                    _row.append(col)
                _table.append(_row)
            yield _table

    '''def attribute_checking(self,input_pdf, text):
        text_out = []
        if input_pdf.startswith('\\'):
            if not self.output_io.getvalue():
                extract_text_to_fp(self.input_file, self.output_io,laparams=LAParams(line_margin=0.18, line_overlap=0.4, all_texts=False),
                                       output_type='html', codec=None)
            else:
                pass
        else:
            if not self.output_io.getvalue():
                with open(input_pdf,'rb') as input:
                    extract_text_to_fp(input, self.output_io,
                                       laparams=LAParams(line_margin=0.18, line_overlap=0.4, all_texts=False),
                                       output_type='html', codec=None)
            else:
                pass
        html = BeautifulSoup(self.output_io.getvalue(), 'html.parser')
        results = html.find_all(
            lambda tag: tag.name == "div" and ' '.join(text.replace('\n', '').split()[:3]) in tag.text.replace('\n',''))
        # results = html.find_all(lambda tag:tag.name == "div" and text.lower() in tag.text.lower().replace('/n',''))  #if data processes via
        if results:
            if 'bold' in str(results[-1]).lower():
                for span in results[-1]:
                    if 'bold' in span['style'].lower():
                        text_out.append(f'<b>{span.text}</b>')
                    if 'bold' not in span['style'].lower():
                        text_out.append(span.text)
                # print(text_out)
                return ''.join(text_out)
            else:
                return None'''

    def find_headings(self,df):
        df_lists = df.values.tolist()
        for df_row in df_lists:
            df_row = [re.sub(r'\<(.*?)\>','',str(value)).strip() for value in df_row]
            print(f'df row cleaned------> {df_row}')
            if 'Unit of Measure' in df_row:
                return df_row

            # for value in df_row:
            #     if value:
            #         if 'Unit of Measure' in value:
            #             print(f'df---row------->{df_row}')
            #             return df_row
            #         else:
            #             return None

    def table_non_table_extraction(self, input_pdf, page_no):  # start -1
        final = {}
        # for index, table in enumerate(get_tables(input_pdf, page_no=page_no)):
        if self.input_pdf_holder:
            input_pdf = self.input_pdf_holder
        else:
            input_pdf = self.get_pdf_file(input_pdf)
        # for index, table in enumerate(self.pdf_to_tables(input_pdf, page_no=int(page_no))):
        for index, table in enumerate(self.pdf_to_docx_to_tables(input_pdf, page_no=int(page_no))):
            nutrition_check_point = 0
            df = pd.DataFrame(table)
            df_original = df.copy()
            rows, columns = df.shape
            if self.is_nutrition_table(re.sub(r'\<(.*?)\>', '', str(df[0][0]).replace('\n',' '))):
                # print(f'nutrition-------->{str(df[0][0])}')
                nutrition_check_point = 1
            if nutrition_check_point == 1:
                print(f'nutrition table ----> {table}')
                reversed = False
                df_header , df_temp = None , None
                for column in range(columns)[:1]:
                    for row in range(rows):
                        if df[column][row]:
                            print('is nutrition table check')
                            if 'Mandatory' in re.sub(r'\<(.*?)\>','',str(df[column][row]).replace('\n','')).strip():
                                print('Table in reversed')
                                reversed = True
                                df_header = df[:row]
                                df_temp = df[row:]
                                df_temp = df_temp.iloc[:, ::-1]
                                df_temp = df_temp.replace(to_replace='', value=np.nan)
                                df_temp = df_temp.dropna(axis=1, how='all')
                                df_header = df_header.replace(to_replace='', value=np.nan)
                                df_header = df_header.dropna(axis=1, how='all')
                                df_header.columns = range(df_header.shape[1])
                                df_temp.columns = range(df_temp.shape[1])
                if reversed:
                    df = df_header.append(df_temp)
                    df.fillna('',inplace=True)
                    rows, columns = df.shape
                nutrition_data = {}
                nutrition_data_original = {}
                _headings = self.find_headings(df)
                if _headings:
                    if 'Unit' in _headings[1]:
                        _headings = _headings[1:]
                    else:
                        _headings = _headings[::-1][1:]
                    print(f'headings------>{_headings}')
                for column in range(columns)[:1]:
                    for row in range(rows):
                        if df[column][row]:
                            print(f'nutrition header 0000000----->{df[column][row]}')
                            nutrition_header = re.sub(r'\((.*?)\)|\[.*?\]|\<(.*?)\>','',str(df[column][row]))
                            nutrition_header = nutrition_header.split('/')[0].strip()
                            key = base('ferrero_header',ferrero_header_model).prediction(get_display(nutrition_header))['output']
                            if key in ['nutrition_table_reference']:
                                lang = classify(nutrition_header)[0]
                                final['nutrition_table_contents'] = [{lang:df[column][row]}]
                            elif key not in ['None', 'nutrition_table_reference', 'header']:
                                # print('nutrition_header_cleaned --->',nutrition_header_cleaned)
                                for col_index in range(columns)[1:]:
                                    if key in nutrition_data:
                                        if df[col_index][row] and df[col_index][row] not in ['N', 'GC', 'P', 'Y']:
                                            nutrition_data[key].append(df[col_index][row])
                                        elif df[col_index][row] == "":
                                            nutrition_data[key].append(df[col_index][row])
                                    else:
                                        if df[col_index][row] and df[col_index][row] not in ['N', 'GC', 'P', 'Y']:
                                            nutrition_data[key] = [df[col_index][row]]
                                        elif df[col_index][row] == "":
                                            nutrition_data[key] = [df[col_index][row]]
                            else:
                                pass
                final_nutrition = {}
                try:
                    print('inside headings')
                    for nutritient, nutri_value in nutrition_data.items():
                        if nutritient != 'None':
                            for index, value in enumerate(nutri_value):
                                if value.strip() and value.strip() not in ['N', 'GC', 'P', 'Y']:
                                    if nutritient in final_nutrition:
                                        if 'Unit' in _headings[index]:
                                            if '%' in value:
                                                final_nutrition[nutritient].append({'PDV': {'en': value}})
                                            else:
                                                final_nutrition[nutritient].append({'Unit': {'en': value}})
                                        elif '%' in _headings[index]:
                                            final_nutrition[nutritient].append({'PDV': {'en': value}})
                                        else:
                                            if '%' in value:
                                                final_nutrition[nutritient].append({'PDV': {'en': value}})
                                            else:
                                                final_nutrition[nutritient].append({'Value': {'en': value}})
                                    else:
                                        if 'Unit' in _headings[index]:
                                            if '%' in value:
                                                final_nutrition[nutritient] = [{'PDV': {'en': value}}]
                                            else:
                                                final_nutrition[nutritient] = [{'Unit': {'en': value}}]
                                        elif '%' in _headings[index]:
                                            final_nutrition[nutritient] = [{'PDV': {'en': value}}]
                                        else:
                                            if '%' in value:
                                                final_nutrition[nutritient] = [{'PDV': {'en': value}}]
                                            else:
                                                final_nutrition[nutritient] = [{'Value': {'en': value}}]
                except:
                    print('else nutritient value')
                    print(nutrition_data)
                    for nutritient, nutri_value in nutrition_data.items():
                        print(f"{nutritient}------{nutri_value}")
                        if nutritient != 'None':
                            for index, value in enumerate(nutri_value):
                                if value.strip() and value.strip() not in ['N', 'GC', 'P', 'Y']:
                                    if nutritient in final_nutrition:
                                        if '%' in value and bool(re.search(r"\d",value)):
                                            final_nutrition[nutritient].append({'PDV': {'en': value}})
                                        elif bool(re.search(r"\d",value)) and not "%" in value:
                                            final_nutrition[nutritient].append({'Value': {'en': value}})
                                        else:
                                            final_nutrition[nutritient].append({'Unit': {'en': value}})
                                    else:
                                        if '%' in value and bool(re.search(r"\d",value)):
                                            final_nutrition[nutritient] = [{'PDV': {'en': value}}]
                                        elif bool(re.search(r"\d",value)) and not "%" in value:
                                            final_nutrition[nutritient] = [{'Value': {'en': value}}]
                                        else:
                                            final_nutrition[nutritient] = [{'Unit': {'en': value}}]
                final['Nutrition'] = final_nutrition
            else:
                # print(f'df------->{df}')
                for column in range(columns)[:1]:
                    for row in range(rows):
                        if df[column][row]:
                            # text = re.sub(r'\((.*?)\)|\[.*?\]|\<(.*?)\>', '', df[column][row].replace('\n', '')).strip()
                            text = re.sub(r'\((.*?)\)|\[.*?\]|\<(.*?)\>', '', df[column][row].replace('\n', '')).strip()
                            cate_out = base('ferrero_header',ferrero_header_model).prediction(text)
                            cate = cate_out['output']
                            cate_probability = cate_out['probability']
                            print(f'{text}--------->{cate}')
                            if cate not in ['None','header','distributor','manufacturer'] and cate_probability > 0.90:
                                for col_index in range(columns)[1:]:
                                    if df[col_index][row] and df[col_index][row] not in ['N', 'GC', 'P', 'Y']:
                                        cleaned_text = re.sub(r'\<(.*?)\>', '', str(df[col_index][row]))
                                        try:
                                            lang = TextBlob(cleaned_text).detect_language()
                                        except:
                                            lang = classify(cleaned_text)[0]
                                        text_final = df[col_index][row]
                                        if cate in ['storage instruction', 'ingredients', 'marketing claim','brand name']:
                                            pred_out = base('general', model_location).prediction(cleaned_text)
                                            pred = pred_out['output']
                                            probability = pred_out['probability']
                                            # print(f'{cleaned_text}------->{pred_out}')
                                            if pred != cate and pred in ['ingredients','storage instruction']:
                                                if probability > 0.80:
                                                    if pred in final:
                                                        final[pred].append({lang: text_final})
                                                    else:
                                                        final[pred] = [{lang: text_final}]
                                                    continue
                                                else:
                                                    pass
                                        if cate in final:
                                            final[cate].append({lang: text_final})
                                        else:
                                            final[cate] = [{lang: text_final}]
                            else:
                                pass
        return final

    def main(self,files,pages):
        for file in [files]:
            out_pages = {}
            for page in pages.split(','):
                response = self.table_non_table_extraction(file,int(page))
                out_pages[page] = response
            # out_file[file] = out_pages
        try:
            self.input_file.close()
            self.output_io.close()
        except:
            pass
        return out_pages
