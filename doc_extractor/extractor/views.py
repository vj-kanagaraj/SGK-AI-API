# Django Libraries
from django.shortcuts import render
from django.http import HttpResponse , JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Rest framework import
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view,permission_classes,authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication , TokenAuthentication

import langdetect as lang_det
from langdetect import DetectorFactory
DetectorFactory.seed = 1

# import other class
# from .excel_processing import *
from .msd_processing import *
from .ferrero_processing import *
from .DG_processing import *
from .GFS_excel_processing import *
from .kellogs_extraction import *
from .Nestle_processing import *
from .carrefour_excel_processing import excel_extract_carrefour
from .General_mills_processing import main as gm_main
from .docx_tag_content_extractor_for_tornado import docx_tag_extractor_for_tornado as docx_ext_tornado

# @api_view()
# @permission_classes([IsAuthenticated])
# @authentication_classes([TokenAuthentication])
def msd(request):
    final_json = {}
    # getting value from query string
    file_name_list = request.GET.getlist('file','no file')
    print('file_list',file_name_list)
    if file_name_list == 'no file':
        return render(request, 'extractor/index_msd.html')
        # return Response({'status':'0'})
    else:
        pass
    for file_index , file_name in enumerate(file_name_list):
        print(f'{file_index}----->{file_name}')
        doc_format = os.path.splitext(file_name)[1].lower()
        if doc_format == '.docx':
            output = msd_extraction().main(file_name)
            final_json[file_index] = output
        else:
            final_json[file_index] = {'status':0,'comment':'file type not supported'}
    return JsonResponse(final_json)

def ferrero(request):
    ferrero_final = {}
    files = request.GET.getlist('file',None)
    pages = request.GET.getlist('pages',None)
    print(f'files---------->{files}')
    print(f'pages---------->{pages}')
    if len(files) == len(pages):
        for index , file in enumerate(files):
            # if file.startswith('\\'):
            #     pass
            # else:
            #     file = document_location + file
            doc_format = os.path.splitext(file)[1].lower()
            if doc_format == '.pdf' and pages:
                out = ferrero_extraction().main(file,pages[index])
                ferrero_final[file] = out
    else:
        ferrero_final = {'status':0,'comment':'please provide correct query strings'}
    return JsonResponse(ferrero_final)

def dollar_general(request):
    files = request.GET.getlist('file',None)
    pages = request.GET.getlist('pages',None)
    final = {}
    results = []
    if len(files) == len(pages):
        for index,file in enumerate(files):
            dg = Dollar_General()
            results.append(dg.main(file,pages[index]))
        for index,result in enumerate(results):
            final[files[index]] = result
    else:
        final = {'status':0,'comment':'Please provide proper query strings'}
    print(final)
    return JsonResponse(final)

def excel_extraction(request):
    output_files = {}
    files = request.GET.getlist('file',None)
    sheet_names = request.GET.getlist('sheet',None)
    print(f'file-------->{files}')
    print(f'sheet_name-------->{sheet_names}')
    for index , file in enumerate(files):
        output_sheets = {}
        for sheet in sheet_names[index].split(','):
            doc_format = os.path.splitext(file)[1].lower()
            if doc_format == '.xlsx' and sheet:
                output = excel_extraction_new(file,sheetname=sheet)
            else:
                output = {'status':0}
            output_sheets[sheet] = output
        output_files[file] = output_sheets
    return JsonResponse(output_files)

def kelloggs_extraction(request):
    output_files = {}
    files = request.GET.getlist('file',None)
    sheet_names = request.GET.getlist('sheet',None)
    print(f'file-------->{files}')
    print(f'sheet_name-------->{sheet_names}')
    for index , file in enumerate(files):
        output_sheets = {}
        for sheet in sheet_names[index].split(','):
            doc_format = os.path.splitext(file)[1].lower()
            if doc_format == '.xlsx' and sheet:
                output = excel_extract_kelloggs(file,sheetname=sheet)
            else:
                output = {'status':0}
            output_sheets[sheet] = output
        # try:
        #     log_book(accounts='Kellogs Excel', input_file=file, input_body={}, output=output_sheets).save()
        # except:
        #     pass
        output_files[file] = output_sheets
    return JsonResponse(output_files)

def nestle(request):
    files = request.GET.getlist('file',None)
    pages = request.GET.getlist('pages',None)
    final = {}
    results = []
    if len(files) == len(pages):
        for index,file in enumerate(files):
            nestle = Nestle_processing()
            results.append(nestle.main(file,pages[index]))
        for index,result in enumerate(results):
            # try:
            #     log_book(accounts='Nestle', input_file=files[index], input_body={}, output=result).save()
            # except:
            #     pass
            final[files[index]] = result
    else:
        final = {'status':0,'comment':'Please provide proper query strings'}
    return JsonResponse(final)

def general_mills_hd(request):
    final = {}
    files = request.GET.getlist('file',None)
    for index , file in enumerate(files):
        doc_format = os.path.splitext(file)[1].lower()
        if doc_format == '.docx':
            result = gm_main(file)
            final[file] = result
        else:
            final[file] = {'status':0,'comment':'please check the file format'}
    return JsonResponse(final)

def carrefour_excel(request):
    output_files = {}
    files = request.GET.getlist('file',None)
    sheet_names = request.GET.getlist('sheet',None)
    print(f'file-------->{files}')
    print(f'sheet_name-------->{sheet_names}')
    for index , file in enumerate(files):
        output_sheets = {}
        for sheet in sheet_names[index].split(','):
            doc_format = os.path.splitext(file)[1].lower()
            if doc_format == '.xlsx' and sheet:
                output = excel_extract_carrefour(file,sheetname=sheet)
            else:
                output = {'status':0,'comment':'please check the input file format'}
            output_sheets[sheet] = output
        # try:
        #     log_book(accounts='Carrefour Excel', input_file=file, input_body={}, output=output_sheets).save()
        # except:
        #     pass
        output_files[file] = output_sheets
    return JsonResponse(output_files)

@csrf_exempt
def language_detection(request):
    langs = set()
    if request.method == 'GET':
        text_list = request.GET.getlist('text',None)
    else:
        text_list = request.POST.getlist('text',None)
    for text in text_list:
        cleaned_text = text.lower().strip()
        cleaned_text = re.sub(r"\d",'',cleaned_text)
        text_array = cleaned_text.split('/')
        for text in text_array:
            text = re.sub(r'[^\w\s]', '', text).strip()
            text = text.replace('\n',' ')
            if text:
                lang = None
                fastext_probability = language_model.predict_pro(text)[0]
                classify_language = classify(text)[0]
                langdetect = lang_det.detect_langs(text)
                print(f'fasttext probability----->{fastext_probability}')
                print(f'classify probability----->{classify_language}')
                print(f'lang detect _probability----->{langdetect}')
                if fastext_probability[1] > 0.70 or fastext_probability[0] in ['en']:
                    lang = fastext_probability[0]
                if (classify_language == fastext_probability[0] and fastext_probability[1] > 0.60) or (classify_language == 'en' and fastext_probability[0] == 'en'):
                    langs.add(classify_language)
                    continue
                for language_probability in langdetect:
                    ld_lang, ld_probability = str(language_probability).split(':')
                    if str(ld_lang).strip() == 'en':
                        lang = 'en'
                        break
                    if float(ld_probability) > 0.75 or str(ld_lang).strip() == str(fastext_probability[0]).strip():
                        lang = str(ld_lang)
                    else:
                        try:
                            lang = TextBlob(text).detect_language()
                        except:
                            lang = langdetect[0].split(':')[0]
                langs.add(lang)
    return HttpResponse(",".join(list(langs)))

def docx_tag_content_extractor_for_tornado(request):
    files = request.GET.getlist('file',None)
    tags = request.GET.getlist('tag',None)
    print(f'filessss------>{files}')
    print(f'tagsss------>{tags}')
    result = None
    for index,file_name in enumerate(files):
        doc_format = os.path.splitext(file_name)[1].lower()
        if 'html' in doc_format.lower():
            doc_type = 'html'
        elif 'xml' in doc_format.lower():
            doc_type = 'xml'
        else:
            raise NotImplementedError('This module is available for html and xml formats')
        result = docx_ext_tornado(input_file=file_name,tags=tags[index],input_type=doc_type).extract()
        print(f'result length------>{len(result)}')
    return JsonResponse({'output': result})


# def dataset_to_mangodb(request):
#     method = request.GET.get('method','fail')
#     migration = request.GET.get('mode','fail')
#     from pymongo import MongoClient
#     client = MongoClient('172.28.42.150',27017)
#     db = client['ai']
#     if method == 'ferrero' and migration == 'dev_to_prod':
#         collection = db['extractor_ferrero_header']
#         data = [ferrero_header(category=i['category'], text=i['text'],) for i in collection.find({})]
#         if data:
#             ferrero_header.objects.all().delete()
#             ferrero_header.objects.bulk_create(data)
#             return HttpResponse('success')
#     elif method == 'general' and migration == 'dev_to_prod':
#         collection = db['extractor_general_dataset']
#         data = [general_dataset(category=i['category'], text=i['text'],) for i in collection.find({})]
#         if data:
#             general_dataset.objects.all().delete()
#             general_dataset.objects.bulk_create(data)
#             return HttpResponse('success')
#     else:
#         return HttpResponse('Failure')
