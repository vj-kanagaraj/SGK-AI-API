import pdfplumber
from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import imutils
import tempfile
from bidi.algorithm import get_display
import camelot
# from camelot import utils

from environment import MODE

if MODE == 'local':
    from .local_constants import *
else:
    from .dev_constants import *

from .excel_processing import *

class mondelez_pdf(object):
    def __init__(self):
        self.input_pdf = None
        self.table_check = ['nutrition information','nutrition information typical values','nutrition declaration']
        self.temp_directory = tempfile.TemporaryDirectory(dir=document_location)
        self.input_pdf_location = f'{self.temp_directory.name}/input_pdf.pdf'
        self.converted_docx = f'{self.temp_directory.name}/converted.docx'
        self.pdfplumber_pdf = None

    def get_input(self,input_pdf):
        if input_pdf.startswith('\\'):
            print('connecting to SMB share')
            try:
                with smbclient.open_file(r"{}".format(input_pdf), mode='rb', username=smb_username,password=smb_password) as f:
                    with open(self.input_pdf_location, 'wb') as pdf:
                        pdf.write(f.read())
                    print('file found')
            except:
                smbclient.reset_connection_cache()
                with smbclient.open_file(r"{}".format(input_pdf), mode='rb', username=smb_username,password=smb_password) as f:
                    with open(self.input_pdf_location, 'wb') as pdf:
                        pdf.write(f.read())
                    print('file found')
            finally:
                smbclient.reset_connection_cache()
            return self.input_pdf_location
        else:
            return document_location+input_pdf

    def pdf_to_image(self):
        images = convert_from_path(self.input_pdf)
        for index, image in enumerate(images):
            image.save(f'{self.temp_directory.name}/{index + 1}.png')
        return 'success'

    def find_contours(self,input_image):
        im = cv2.imread(input_image)
        height = im.shape[0]
        width = im.shape[1]
        # de_img = cv2.GaussianBlur(im, (7, 7), 0)
        gray_scale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray_scale[gray_scale < 250] = 0
        th1, img_bin = cv2.threshold(gray_scale, 150, 225, cv2.THRESH_BINARY)
        img_bin = ~img_bin
        line_min_width_horizontal = 50
        line_min_width_vertical = 30
        kernal_h = np.ones((1, line_min_width_horizontal), np.uint8)
        kernal_v = np.ones((line_min_width_vertical, 1), np.uint8)
        img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
        img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
        img_bin_final = img_bin_h | img_bin_v
        final_kernel = np.ones((3, 3), np.uint8)
        img_bin_final_dilation = cv2.dilate(img_bin_final, final_kernel, iterations=1)
        can_img = cv2.Canny(img_bin_final_dilation, 8, 200, 100)
        cnts = cv2.findContours(can_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts)
        cnts2 = [cnt for cnt in cnts1 if cv2.contourArea(cnt) > 5000]
        i = 0
        for contour in cnts2:
            if cv2.contourArea(contour) > 4000:
                x, y, w, h = cv2.boundingRect(contour)
                i = i + 1
                yield (width / (x - 10), height / (y - 10), width / (x + w + 20), height / (y + h + 30))

    def content_inside_bounding_box(self, page_no, coordinates_percent):
        pdf = pdfplumber.open(self.input_pdf)
        page = pdf.pages[page_no - 1]
        pages = len(pdf.pages)  # getting total pages
        height, width = float(page.height), float(page.width)
        # layout, dim = utils.get_page_layout(self.input_pdf)
        w0, h0, w1, h1 = coordinates_percent
        coordinates = (width / w0, height / h0, width / w1, height / h1)
        # x1,y1,x2,y2 = coordinates
        ROI = page.within_bbox(coordinates, relative=False)
        table_custom = ROI.extract_tables(
            table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 4})
        table_normal = ROI.extract_tables()
        # try:
        #     camelot_table = camelot.read_pdf(self.input_pdf,table_regions=[f'{x1},{y1},{x2},{y2}'],pages=str(page_no))
        #     print('camelot_table---->',camelot_table[0].df)
        # except:
        #     pass
        if table_normal and table_custom:
            table_custom_shape = pd.DataFrame(table_custom[0]).shape[1]
            table_normal_shape = pd.DataFrame(table_normal[0]).shape[1]
            if table_normal_shape == table_custom_shape:
                table = table_custom
            elif table_normal_shape > table_custom_shape:
                table = table_normal
            else:
                table = table_custom
            yield (table, 'table')
        elif table_normal and not table_custom:
            table = table_normal
            yield (table, 'table')
        elif table_custom and not table_normal:
            table = table_custom
            yield (table, 'table')
        else:
            content = ROI.extract_text()
            yield (content, 'content')

    def is_nutrition_table_or_not(self,text):
        similarity = 0
        if isinstance(text,str):
            similarity = cosine_similarity(laser.embed_sentences(text, lang='en'),
                                           laser.embed_sentences(self.table_check, lang='en').mean(0).reshape(1, 1024))[0][0]
            print(text,'----->', similarity)
            if similarity > 0.80:
                return True
            else:
                return False
        elif isinstance(text,list):
            text = [t for t in text if t.strip()][::-1]
            similarity_check = lambda x : cosine_similarity(laser.embed_sentences(x, lang='en'),
                                           laser.embed_sentences(self.table_check, lang='en').mean(0).reshape(1, 1024))[0][0]
            print(text,'----->', similarity)
            print('similarity check------->',similarity_check(text[-1]))
            if any(similarity_check(t.replace('\n','')) > 0.80 for t in text if t.strip()):
                return True
            else:
                return False

    def is_nutrition_data(self,data):
        sample_text = ['Fat   \nOf which saturates \nCarbohydrate    \nof which sugars   \nFibre   \nProtein   \nSalt   \nEnergy   \nCalories']
        similarity = cosine_similarity(laser.embed_sentences(data, lang='en'),
                                       laser.embed_sentences(sample_text, lang='en').mean(0).reshape(1, 1024))[0][0]
        print('Nutrition data check =======>',similarity)
        if similarity > 0.80:
            return True
        else:
            return False


    def mondelez_classifier(self,text,method=None):
        import os
        if method == 'General':
            model_location = mondelez_pdf_general_model_location
            dataset_location = mondelez_dataset
            if os.path.exists(model_location):
                classifier = joblib.load(model_location)
            else:
                dataframe = pd.read_excel(dataset_location, sheet_name='Sheet2',engine='openpyxl')
                x_train_laser = laser.embed_sentences(dataframe['text'], lang='en')
                classifier = MLPClassifier(hidden_layer_sizes=(80,), solver='adam', activation='tanh', max_iter=750,
                                           random_state=0, shuffle=True)
                classifier.fit(x_train_laser, dataframe['category'])
                joblib.dump(classifier, model_location)
        else:
            model_location = mondelez_pdf_model_location
            dataset_location = mondelez_dataset
            if os.path.exists(model_location):
                classifier = joblib.load(model_location)
            else:
                dataframe = pd.read_excel(dataset_location, engine='openpyxl')
                x_train_laser = laser.embed_sentences(dataframe['text'], lang='en')
                classifier = MLPClassifier(hidden_layer_sizes=(80,), solver='adam', activation='tanh', max_iter=750,
                                           random_state=0, shuffle=True)
                classifier.fit(x_train_laser, dataframe['category'])
                joblib.dump(classifier, model_location)
        prediction = classifier.predict(laser.embed_sentences([text], lang='en'))
        probability = classifier.predict_proba(laser.embed_sentences([text], lang='en'))
        probability[0].sort()
        max_probability = max(probability[0])
        if max_probability > 0.50:
            pred_output = prediction[0]
        else:
            pred_output = 'None'
        # print('*****'*5)
        # print(text)
        # print(pred_output,'------>',max_probability)
        # print('*****'*5)
        return {'probability': max_probability, 'output': pred_output}

    def nutrition_table_processing(self,page_no, table_type=None):
        nutrition_dict = {}
        tables = camelot.read_pdf(self.input_pdf, pages=str(page_no), flavor='stream', row_tol=9)
        no_of_tables = len(tables)
        for table_no in range(no_of_tables):
            df = tables[table_no].df
            # if self.is_nutrition_table_or_not(str(df[0][0]).split('/')[0]):
            if self.is_nutrition_table_or_not(df.loc[0].to_list()):
                rows , columns = df.shape
                for column in range(columns)[:1]:
                    for row in range(rows)[1:]:
                        if table_type == 'Normal':
                            nutrition_header = str(df[column][row]).strip().lower()
                            if nutrition_header:
                                nutrition_header = str(df[column][row]).split('/')[0]
                            elif any(('kcal' in str(df[_col][row]).lower() for _col in range(columns)[column + 1:] if str(df[_col][row]).strip())):
                                nutrition_header = 'calories'
                            else:
                                continue
                            nutrition_output = base('ferrero_header', ferrero_header_model).prediction(get_display(nutrition_header))
                            if nutrition_output['output'] not in ['None'] and nutrition_output['probability'] > 0.20 :
                                for _col in range(columns)[column+1:]:
                                    value = str(df[_col][row]).strip()
                                    if value:
                                        value_header = 'PDV' if "%" in value else "Value"
                                        if nutrition_output['output'] in nutrition_dict:
                                            nutrition_dict[nutrition_output['output']].append({value_header:{'en':value}})
                                        else:
                                            nutrition_dict[nutrition_output['output']] = [{value_header:{'en':value}}]
                        else:
                            header = str(df[column][row]).strip()
                            header = re.sub(r"\(.*\)", '', header)
                            if header and 'serving' not in header.lower():
                                nutrition_header = re.findall(r"^\n?([A-Za-z].*)\s?\/", header, re.I | re.MULTILINE)
                                # print('nutrition_header----->', nutrition_header)
                                if nutrition_header:
                                    nutrition_header = nutrition_header[0]
                                    value = re.findall(r"(\<?s?\d?\.?\d{1,2}\s?(g|kj|kcal|mg|mcg))",str(df[column][row]), re.I)
                                    # print('value------>', value)
                                    if value:
                                        value_header_regex = "PDV" if "%" in value else "Value"
                                        value = value[0][0]
                                        if value.strip():
                                            if nutrition_header in nutrition_dict:
                                                nutrition_dict[nutrition_header].append({value_header_regex: {'en': value}})
                                            else:
                                                nutrition_dict[nutrition_header] = [{value_header_regex: {'en': value}}]
                                    for _col in range(columns)[column + 1:]:
                                        value = str(df[_col][row]).strip()
                                        value_header = 'PDV' if "%" in value else "Value"
                                        if value:
                                            if nutrition_header in nutrition_dict:
                                                nutrition_dict[nutrition_header].append({value_header: {'en': value}})
                                            else:
                                                nutrition_dict[nutrition_header] = [{value_header: {'en': value}}]
        # print('Nutrition_dictionary---->',nutrition_dict)
        return nutrition_dict

    def normal_table_processing(self,df):
        normal_dict = {}
        rows , columns = df.shape
        if columns == 1:
            print('inside column 1 ---- normal table')
            for column in range(columns)[:1]:
                for row in range(rows):
                    header = str(df[column][row]).strip()
                    # cleaned_header = re.sub(r"\(.*\)",'',header)
                    # cleaned_header = cleaned_header.split('/')[0]
                    classifier_output = self.mondelez_classifier(get_display(header),method='General')
                    # lang = classify(header)[0]
                    try:
                        lang = lang_detect(header)
                    except:
                        lang = classify(header)[0]
                    print('detected sentence---->', header)
                    print(classifier_output['output'], '------>', classifier_output['probability'])
                    # classifier_output = base('general', model_location).prediction(get_display(cleaned_header))
                    if classifier_output['output'] in ['INGREDIENTS_DECLARATION','ALLERGEN_STATEMENT'] and classifier_output['probability'] > 0.70:
                        # print('detected sentence---->', header)
                        header = re.sub(r"(\s?\n\s{0,1}){2,5}\n?","\n",header)
                        if 'INGREDIENTS_DECLARATION' in normal_dict:
                            normal_dict['INGREDIENTS_DECLARATION'].append({lang: header})
                        else:
                            normal_dict['INGREDIENTS_DECLARATION'] = [{lang: header}]
                    else:
                        if 'unmapped' in normal_dict:
                            normal_dict['unmapped'].append({lang: header})
                        else:
                            normal_dict['unmapped'] = [{lang: header}]
        else:
            print('more than one column')
            # print(df)
            for column in range(columns)[:1]:
                for row in range(rows):
                    header = str(df[column][row]).strip()
                    cleaned_header = re.sub(r"\(.*\)",'',header)
                    cleaned_header = cleaned_header.split('/')[0]
                    classifier_output = self.mondelez_classifier(get_display(cleaned_header))
                    print('result------->',classifier_output,'----->',cleaned_header)
                    if header:
                        if classifier_output['output'] not in ['None'] and classifier_output['probability'] > 0.80:
                            if classifier_output['output'] in ['BRAND_NAME','VARIANT','FUNCTIONAL_NAME','NET_CONTENT_STATEMENT','LOCATION_OF_ORIGIN','SERVING_SIZE']:
                                for _col in range(columns)[column+1:]:
                                    content = df[_col][row]
                                    if isinstance(content,str) and str(content).strip():
                                        if re.search(r"(\b[A-Z]{2}:)",content,re.M):
                                            # print(content)
                                            content = re.sub(r"\b(([A-Z]{2})?\/?[A-Z]{2}:)",lambda pat: "**"+pat.group(1),content,re.MULTILINE)
                                            # print(content)
                                            for splitted in content.split('**'):
                                                splitted = splitted.strip()
                                                if splitted and isinstance(splitted,str):
                                                    try:
                                                        lang = lang_detect(splitted)
                                                    except:
                                                        lang = classify(splitted)[0]
                                                    if classifier_output['output'] in normal_dict:
                                                        normal_dict[classifier_output['output']].append({lang:splitted})
                                                    else:
                                                        normal_dict[classifier_output['output']] = [{lang:splitted}]
                                        elif re.search(r"\.\s",content):
                                            for splitted in content.split('.'):
                                                splitted = splitted.strip()
                                                if splitted and isinstance(splitted,str):
                                                    try:
                                                        lang = lang_detect(splitted)
                                                    except:
                                                        lang = classify(splitted)[0]
                                                    if classifier_output['output'] in normal_dict:
                                                        normal_dict[classifier_output['output']].append({lang:splitted})
                                                    else:
                                                        normal_dict[classifier_output['output']] = [{lang:splitted}]
                                        else:
                                            for splitted in content.split('\n'):
                                                splitted = splitted.strip()
                                                if splitted and isinstance(splitted,str):
                                                    try:
                                                        lang = lang_detect(splitted)
                                                    except:
                                                        lang = classify(splitted)[0]
                                                    if classifier_output['output'] in normal_dict:
                                                        normal_dict[classifier_output['output']].append({lang:splitted})
                                                    else:
                                                        normal_dict[classifier_output['output']] = [{lang:splitted}]

                            else:
                                for _col in range(columns)[column+1:]:
                                    content = df[_col][row]
                                    if isinstance(content,str) and str(content).strip():
                                        try:
                                            lang = lang_detect(content)
                                        except:
                                            lang = classify(content)[0]
                                        if classifier_output['output'] in normal_dict:
                                            normal_dict[classifier_output['output']].append({lang:content})
                                        else:
                                            normal_dict[classifier_output['output']] = [{lang:content}]
                        elif self.is_nutrition_data(header):            # Nutrition data processing
                            header_column = header.split('\n')
                            overall_columns = [header_column]
                            value_column = []
                            for _col in range(columns)[column + 1:]:
                                content = df[_col][row]
                                if str(content).strip() and isinstance(content,str):
                                    value_column_content = content.split('\n')
                                    value_column_content = [value.strip() for value in value_column_content]
                                    if len(value_column_content) == len(header_column):
                                        value_column.append(value_column_content)
                                    else:
                                        value_column.clear()
                                        break
                            if value_column:
                                overall_columns.extend(value_column)
                                # print('overall_columns--->',overall_columns)
                                df = pd.DataFrame(overall_columns).transpose()
                                rows, columns = df.shape
                                nutrition_dict = {}
                                for column in range(columns)[:1]:
                                    for row in range(rows)[1:]:
                                        nutrition_header = str(df[column][row]).strip().lower()
                                        if nutrition_header:
                                            nutrition_header = str(df[column][row]).split('/')[0]
                                        elif any(('kcal' in str(df[_col][row]).lower() for _col in
                                                  range(columns)[column + 1:] if str(df[_col][row]).strip())):
                                            nutrition_header = 'calories'
                                        else:
                                            continue
                                        nutrition_output = base('ferrero_header', ferrero_header_model).prediction(
                                            get_display(nutrition_header))
                                        if nutrition_output['output'] not in ['None'] and nutrition_output[
                                            'probability'] > 0.20:
                                            for _col in range(columns)[column + 1:]:
                                                value = str(df[_col][row]).strip()
                                                if value:
                                                    value_header = 'PDV' if "%" in value else "Value"
                                                    if nutrition_output['output'] in nutrition_dict:
                                                        nutrition_dict[nutrition_output['output']].append(
                                                            {value_header: {'en': value}})
                                                    else:
                                                        nutrition_dict[nutrition_output['output']] = [
                                                            {value_header: {'en': value}}]
                                if 'NUTRITION_FACTS' in normal_dict:
                                    normal_dict['NUTRITION_FACTS'].append(nutrition_dict)
                                else:
                                    normal_dict['NUTRITION_FACTS'] = [nutrition_dict]
                    else:
                        for _col in range(columns)[column + 1:]:
                            content = df[_col][row]
                            if isinstance(content,str) and content.strip():
                                classifier_output = self.mondelez_classifier(get_display(content),method='General')
                                # lang = classify(content)[0]
                                try:
                                    lang = lang_detect(content)
                                except:
                                    lang = classify(content)[0]
                                if classifier_output['output'] in ['INGREDIENTS_DECLARATION', 'ALLERGEN_STATEMENT'] and classifier_output['probability'] > 0.70:
                                    if 'INGREDIENTS_DECLARATION' in normal_dict:
                                        normal_dict['INGREDIENTS_DECLARATION'].append({lang: content})
                                    else:
                                        normal_dict['INGREDIENTS_DECLARATION'] = [{lang: content}]
                                else:
                                    if 'unmapped' in normal_dict:
                                        normal_dict['unmapped'].append({lang: content})
                                    else:
                                        normal_dict['unmapped'] = [{lang: content}]
        return normal_dict

    def main(self,input_pdf,pages):
        final_dict = {}
        self.input_pdf = self.get_input(input_pdf)
        self.pdfplumber_pdf = pdfplumber.open(self.input_pdf)
        pdf_to_image_status = self.pdf_to_image()
        assert pdf_to_image_status == 'success', 'pdf to image conversion failed'
        for page in pages.split(','):
            print(f'{page}')
            if int(page)-1 in range(len(self.pdfplumber_pdf.pages)):
                page_dict = {}
                input_image = f'{self.temp_directory.name}/{page}.png'
                for bounding_box in self.find_contours(input_image):
                    for content, type in self.content_inside_bounding_box(int(page), bounding_box):
                        print(type)
                        if type == 'table':
                            df = pd.DataFrame(content[0])
                            table_heading = str(df[0][0]).split('/')
                            # print('table_heading', table_heading)
                            table_heading = [split_heading for split_heading in table_heading if split_heading.strip()]
                            # print('table_heading', table_heading)
                            type = 'Normal' if len(table_heading) == 1 else 'Arabic'
                            if table_heading:
                                if self.is_nutrition_table_or_not(table_heading[0].strip().split('\n')[0]):
                                    print('inside nutrition table--------->',type)
                                    # need to pass to nutrition table processing
                                    nutrition_dict = self.nutrition_table_processing(page_no=page,table_type=type)
                                    if 'NUTRITION_FACTS' in page_dict:
                                        page_dict['NUTRITION_FACTS'].append(nutrition_dict)
                                    else:
                                        page_dict['NUTRITION_FACTS'] = [nutrition_dict]
                                    # print('Nutrition----->',nutrition_dict)
                                else:
                                    normal_dict = self.normal_table_processing(df)
                                    page_dict = {**page_dict,**normal_dict}
                            else:
                                print('inside fault region')
                                normal_dict = self.normal_table_processing(df)
                                page_dict = {**page_dict, **normal_dict}
                                # rows , columns = df.shape
                                # for column in range(columns)[1:]:
                                #     for row in range(rows):
                                #         value = str(df[column][row]).strip()
                                #         if value:
                                #             lang = classify(value)[0]
                                #             if 'unmapped' in page_dict:
                                #                 page_dict['unmapped'].append({lang: value})
                                #             else:
                                #                 page_dict['unmapped'] = [{lang: value}]
                        elif type == 'content':
                            if isinstance(content,str):
                                # classifier_output = base('general', model_location).prediction(get_display(content))
                                classifier_output = self.mondelez_classifier(get_display(content),method='General')
                                # lang = classify(content)[0]
                                try:
                                    lang = lang_detect(content)
                                except:
                                    lang = classify(content)[0]
                                print('detected sentence---->', content)
                                print(classifier_output['output'],'------>', classifier_output['probability'])
                                if classifier_output['output'] in ['INGREDIENTS_DECLARATION','ALLERGEN_STATEMENT'] and classifier_output['probability'] > 0.70:  # can reduce the probability score
                                    if 'INGREDIENTS_DECLARATION' in page_dict:
                                        page_dict['INGREDIENTS_DECLARATION'].append({lang:content})
                                    else:
                                        page_dict['INGREDIENTS_DECLARATION'] = [{lang:content}]
                                elif classifier_output['output'] in ['NUTRITION_FACTS'] and classifier_output['probability'] > 0.70:
                                    nutrition_content = content.lower()
                                    if 'nutrition facts' in nutrition_content and ':' in nutrition_content:
                                        nutrition_content_dict = {}
                                        _content = nutrition_content.split(':')
                                        # print(_content)
                                        _content = _content[-1]
                                        # print(_content)
                                        _content_list = _content.split(',')
                                        # print(_content)
                                        for row_element in _content_list:
                                            row_element = re.sub(r"[^0-9A-Za-z\%\s\.]", "", row_element)
                                            row_element = re.sub(r"\s{1,4}", " ", row_element)
                                            extracted_list_tuple = re.findall(r"(\D*)\s?(\<?s?\d{1,2}?\.?\d{1,2}\s?(g|kj|kcal|mg|mcg))\s?(\<?s?\d?\.?\d{1,2}\s?%)",row_element)      # eg : [('Saturated Fat ', '16 g', 'g', '8 %')]
                                            if extracted_list_tuple:
                                                extracted_list = list(extracted_list_tuple[0])
                                                if len(extracted_list) == 4:
                                                    del extracted_list[2]
                                                    header = str(extracted_list[0]).strip().capitalize()
                                                    for value in extracted_list[1:]:
                                                        value_header = 'PDV' if "%" in value else "Value"
                                                        if value:
                                                            if header in nutrition_content_dict:
                                                                nutrition_content_dict[header].append({value_header: {'en': value}})
                                                            else:
                                                                nutrition_content_dict[header] = [{value_header: {'en': value}}]
                                        if 'NUTRITION_FACTS' in page_dict:
                                            page_dict['NUTRITION_FACTS'].append(nutrition_content_dict)
                                        else:
                                            page_dict['NUTRITION_FACTS'] = [nutrition_content_dict]
                                    else:
                                        if 'NUTRITIONAL_CLAIM' in page_dict:
                                            page_dict['NUTRITIONAL_CLAIM'].append({lang: content})
                                        else:
                                            page_dict['NUTRITIONAL_CLAIM'] = [{lang: content}]
                                elif 'nutrition' in content.lower() and 'template' in content.lower():
                                    if 'Nutrition Declaration' in page_dict:
                                        page_dict['Nutrition Declaration'].append({lang: content})
                                    else:
                                        page_dict['Nutrition Declaration'] = [{lang: content}]
                                else:
                                    # print('non detected sentence---->',content)
                                    if 'unmapped' in page_dict:
                                        page_dict['unmapped'].append({lang: content})
                                    else:
                                        page_dict['unmapped'] = [{lang: content}]
                final_dict[page] = page_dict
        return final_dict

