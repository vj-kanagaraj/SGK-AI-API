import mammoth
from bs4 import BeautifulSoup
from functools import partial
from bidi.algorithm import get_display
from .excel_processing import *
from sklearn.metrics.pairwise import cosine_similarity


# GM_HD_model_dataset = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/GM_HD_headers_dataset.xlsx"
# GM_HD_model_location = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/GM_HD_model.pkl"
# ferrero_model_location = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/ferrero_header_model.pkl"

# ip_docx = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/HD_M7 Belgian Chocolate Hazelnut 100ml_PS 8261249 ACS0220_kl1.docx"
# ip_docx = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/HD_M7 Dark Chocolate & Caramelized Almonds 100ml_PS 8261255 ACS0220_kl1.docx"
# ip_docx = r"/Users/VIJAYKANAGARAJ/PycharmProjects/Schawk_document_xml/Dataset/HD_ P2 Dark Chocolate & Caramelized Almond 460ML_PS 8251496 ACS0620_KL1.docx"


header_memory = ""

def docx_to_html(file):
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

def docx_to_table(input_file):
    html = docx_to_html(input_file)
    # html = mammoth.convert_to_html(input_file).value
    soup = BeautifulSoup(html,"html.parser")
    for tables in soup.find_all('table'):
        for table in tables:
            row_values = []
            for row in table.find_all('tr'):
                column_values = []
                for column in row.find_all('td'):
                    if column.text.strip():
                        column_values.append(column.text)
                else:
                    if column_values:
                        row_values.append(column_values)
            else:
                if row_values:
                    df = pd.DataFrame(row_values)
                    # if df.shape[0] < 15:
                    if df.shape[0] < 10:
                        print('---' * 10)
                        yield row_values

def classifier(model_location,text):
    if os.path.exists(model_location):
        model = joblib.load(model_location)
    else:
        print('model training')
        df = pd.read_excel(GM_HD_model_dataset, engine='openpyxl')
        df = df.sample(frac=1)
        list_text = df['text'].tolist()
        preprocessed_text = []
        for text in list_text:
            preprocessed_text.append(text_preprocessing(text))
        df['cleaned_text'] = preprocessed_text
        X_train_laser = laser.embed_sentences(df['cleaned_text'], lang='en')
        model = MLPClassifier(hidden_layer_sizes=(80,), solver='adam', activation='tanh', max_iter=750,
                              random_state=0, shuffle=True)
        model.fit(X_train_laser, df['category'])
        joblib.dump(model, model_location)
    prediction = model.predict(laser.embed_sentences([text], lang='en'))
    probability = model.predict_proba(laser.embed_sentences([text], lang='en'))
    probability[0].sort()
    max_probability = max(probability[0])
    if max_probability > 0.65:
        pred_output = prediction[0]
    else:
        pred_output = 'None'
    # print(text)
    # print({'probability': max_probability, 'output': pred_output})
    # print('----------'*5)
    return {'probability': max_probability, 'output': pred_output}

def GM_header_classifier(model_location,model_dataset,text):
    if os.path.exists(model_location):
        model = joblib.load(model_location)
    else:
        print('model training')
        df = pd.read_excel(model_dataset, engine='openpyxl')
        df = df.sample(frac=1)
        list_text = df['text'].tolist()
        preprocessed_text = []
        for text in list_text:
            preprocessed_text.append(text_preprocessing(text))
        df['cleaned_text'] = preprocessed_text
        X_train_laser = laser.embed_sentences(df['cleaned_text'], lang='en')
        model = MLPClassifier(hidden_layer_sizes=(80,), solver='adam', activation='tanh', max_iter=750,
                              random_state=0, shuffle=True)
        model.fit(X_train_laser, df['category'])
        joblib.dump(model, model_location)
    prediction = model.predict(laser.embed_sentences([text], lang='en'))
    probability = model.predict_proba(laser.embed_sentences([text], lang='en'))
    probability[0].sort()
    max_probability = max(probability[0])
    if max_probability > 0.65:
        pred_output = prediction[0]
    else:
        pred_output = 'None'
    # print(text)
    # print({'probability': max_probability, 'output': pred_output})
    # print('----------'*5)
    return {'probability': max_probability, 'output': pred_output}

def text_preprocessing(text):
    text = str(text)
    text = text.lower()
    text = text.replace('\r','\n')
    # text = re.sub(r'[^\w\s]','',text)
    # text = re.sub(r"\(.*\)|\[.*\]","",text)
    text = text.replace('(','').replace(')','')
    text = re.sub(r"\[.*\]","",text)
    return text

# def is_nutrition_table(table : list) -> str:
#     nutrition_table_detector = partial(classifier,griesson_model_location)
#     table_text = " ".join([text.strip() for list in table for text in list])
#     # output = base('general', model_location).prediction(get_display(table_text))['output']
#     output = nutrition_table_detector(table_text)['output']
#     return output

def is_nutrition_table_or_not(text):
    table_check = ['nutrition information', 'nutrition information typical values', 'nutrition declaration','Valeurs nutritionnelles moyennes']
    similarity = 0
    if isinstance(text,str):
        similarity = cosine_similarity(laser.embed_sentences(text, lang='en'),
                                       laser.embed_sentences(table_check, lang='en').mean(0).reshape(1, 1024))[0][0]
        # print('*******' * 5)
        print(text, '----->', similarity)
        # print('*******' * 5)
        if similarity > 0.80:
            return True
        else:
            return False

# def nutrition_table_processing_old(table:list) -> dict:
#     nutrition_dict = {}
#     df = pd.DataFrame(table)
#     rows, columns = df.shape
#     if 'nutrition information' in str(df[0][0]).lower() and "rdt" in str(df[0][0]).lower():
#         for column in range(columns)[:1]:
#             for row in range(rows):
#                 nutrition = str(df[column][row]).split('/')[0].strip()
#                 print(f"data--------->{nutrition}")
#                 nutrition_detection = partial(classifier,ferrero_model_location)
#                 output = nutrition_detection(get_display(nutrition))
#                 key = output['output']
#                 probability = output['probability']
#                 print(f'{nutrition}-------->{key}----------->{probability}')
#                 if key not in ['None','nutrition_table_reference']:
#                     for _col in range(columns)[1:]:
#                         nutri_value = df[_col][row]
#                         if nutri_value:
#                             nutri_value = str(nutri_value).strip()
#                             if key in nutrition_dict:
#                                 if '%' in str(df[_col][row]):
#                                     nutrition_dict[key].append({'PDV':{'en':nutri_value}})
#                                 else:
#                                     nutrition_dict[key].append({'Value':{'en':nutri_value}})
#                             else:
#                                 if '%' in str(df[_col][row]):
#                                     nutrition_dict[key] = [{'PDV':{'en':nutri_value}}]
#                                 else:
#                                     nutrition_dict[key] = [{'Value':{'en':nutri_value}}]
#         return  nutrition_dict

def nutrition_table_processing(table:list) -> dict:
    nutrition_dict = {}
    df = pd.DataFrame(table)
    rows, columns = df.shape
    # print('inside nutrition table processing')
    # if 'nutrition information' in str(df[0][0]).lower() and "rdt" in str(df[0][0]).lower():
    if is_nutrition_table_or_not(df[0][0]):
        print('nutrition_table_found')
        for column in range(columns)[:1]:
            for row in range(rows):
                nutrition_original = str(df[column][row])
                nutrition = nutrition_original.split('/')[0].strip()
                print(f"data--------->{nutrition}")
                # nutrition_detection = partial(classifier,ferrero_model_location)
                # output = nutrition_detection(get_display(nutrition))
                # output = base('ferrero_header', ferrero_header_model).prediction(get_display(nutrition),method='labse')
                output = base('ferrero_header', ferrero_header_model).prediction(get_display(nutrition))
                key = output['output']
                probability = output['probability']
                print(f'{nutrition}-------->{key}----------->{probability}')
                if key in ['nutrition_table_reference']:
                    nutrition_dict[key] = {'en': nutrition_original}
                elif key not in ['None','nutrition_table_reference','header']:
                    remaining_cells_in_row = df.loc[row][1:].to_list()
                    remaining_cell_join_text = " ".join((text for text in remaining_cells_in_row if text))
                    # print(f"listttttt join--------->{remaining_cell_join_text}")
                    for value in re.finditer(r"\d{0,4}?\.?\d{0,2}?\D{0,4}?\s?\(?(mg|g|kcal|kj|%|-)\)?\/?(mg|g|kcal|kj|%|-)?",remaining_cell_join_text,re.I):
                    #for value in re.finditer(r"\d{0,4}?\.?\d{0,2}?\s?(mg|g|kcal|kj|%|-)\/?(mg|g|kcal|kj|%|-)?",remaining_cell_join_text,re.I):
                        value = str(value.group()).strip()
                        if value:
                            if key in nutrition_dict:
                                if '%' in str(value):
                                    nutrition_dict[key].append({'PDV':{'en':value}})
                                else:
                                    nutrition_dict[key].append({'Value':{'en':value}})
                            else:
                                if '%' in str(value):
                                    nutrition_dict[key] = [{'PDV':{'en':value}}]
                                else:
                                    nutrition_dict[key] = [{'Value':{'en':value}}]
        # print(nutrition_dict)
        return  nutrition_dict

def normal_table_processing(table:list) -> dict:
    global header_memory
    print('inside normal table processing')
    normal_dict = {}
    df = pd.DataFrame(table)
    print(df)
    rows, columns = df.shape
    if columns > 1 and columns <= 10:
        for column in range(columns)[:1]:
            for row in range(rows):
                # print(df[column][row])
                header = str(df[column][row])
                if re.search(r"^\(.*\)$",header.strip()) and header_memory and header_memory != 'None':
                    header = header_memory
                    print(header,'------->',df)
                header_cleaned = re.sub(r"\(.*\)","",header).strip()
                normal_detection = partial(GM_header_classifier,GM_HD_model_location,GM_HD_model_dataset)
                if header_cleaned:
                    output = normal_detection(get_display(header_cleaned))
                    output_class = output['output']
                    header_memory = header
                    probability = output['probability']
                    if output_class and probability > 0.90:
                        print(f'{header_cleaned}-----{output_class}-------{probability}')
                        # if output_class == "NET_CONTENT_STATEMENT":
                        #     print(table)
                        #     print('----' * 5)
                        if str(df[column+1][row]).strip():
                            value = df[column+1][row]
                            if isinstance(value,str):
                                value = str(df[column+1][row]).strip()
                                value = text_cleaning(value).strip()
                                if value:
                                    try:
                                        lang = lang_detect(value)
                                    except:
                                        lang = classify(value)[0]
                                    if output_class in normal_dict:
                                        normal_dict[output_class].append({lang: value})
                                    else:
                                        normal_dict[output_class] = [{lang: value}]
    return normal_dict

def main(input_docx):
    final_dict = {}
    nutrition_final_dict = {}
    for table in docx_to_table(input_docx):
        # output = is_nutrition_table(table)
        print('first_object------->',table[0][0])
        # output = is_nutrition_table_or_not(table[0][0])
        # if output == "Nutrition":              # Nutrition table processing
        if is_nutrition_table_or_not(table[0][0]):              # Nutrition table processing
            nutrition_information = nutrition_table_processing(table)
            print(nutrition_information)
            if nutrition_information:
                if 'nutrition_table_reference' in nutrition_information:
                    if 'nutrition_table_reference' in final_dict:
                        final_dict['nutrition_table_reference'].append(nutrition_information['nutrition_table_reference'])
                    else:
                        final_dict['nutrition_table_reference'] = [nutrition_information['nutrition_table_reference']]
                    nutrition_information.pop('nutrition_table_reference',None)
                if 'NUTRITION_FACTS' in nutrition_final_dict:
                    nutrition_final_dict['NUTRITION_FACTS'].append(nutrition_information)
                else:
                    nutrition_final_dict['NUTRITION_FACTS'] = [nutrition_information]
        else:
            normal_dictionary = normal_table_processing(table)
            for key , value in normal_dictionary.items():
                if key in final_dict:
                    final_dict[key].append(value[0])
                else:
                    final_dict[key] = value
    # print(final_dict)
    # remove duplicates
    final_cleaned_dict = {}
    for category , value_list in final_dict.items():
        final_cleaned_dict[category] = list({frozenset(list_element.items()) : list_element for list_element in value_list}.values())
    # print('after cleaning---->',final_cleaned_dict)
    return {**nutrition_final_dict,**final_cleaned_dict}

def text_cleaning(text):
    unwanted_text = ['NA','N/A','English Back Translation:','English Back Translation:n/a','English Back Translation:NA',
                     'new EU recycling logo.PNG< a >','English Back Translation:see english sentence','English Back Translation:see french field',
                     'Green dot registered.png< a >',' new EU recycling logo.PNG< a >','English Back Translation:see lead country','English Back Translation:See artwork',
                     'G17 Chocolate 2047032[1]-MGO-G531997A-2','NFP HD CHOCOLATE (MC)','M15 Chocolate - Nutri Info','GM logo.png< a >',
                     'tbc*','address^','icone vegetarienne.PNG< a >','*only new logo eco emballage in yellow box*','*only new logo eco emballage with following text in yellow box*',
                     'new EU recycling logo.PNG< a >','OR (if lack of space )','NA.bmp< a >','*Note: website already in the Generic English Information',
                    ]

    text = re.sub(r"\^.*\^","",text)
    for text_pattern in unwanted_text:
        # text = re.sub(r"{}".format(text_pattern),'',text)
        text = text.replace(text_pattern,'')
    return text
