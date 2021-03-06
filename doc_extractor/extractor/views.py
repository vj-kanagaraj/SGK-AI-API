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

# import other class
# from .excel_processing import *
from .msd_processing import *
from .ferrero_processing import *
from .GFS_excel_processing import *

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
            if file.startswith('\\'):
                pass
            else:
                file = document_location + file

            doc_format = os.path.splitext(file)[1].lower()
            if doc_format == '.pdf' and pages:
                out = ferrero_extraction().main(file,pages[index])
                ferrero_final[file] = out
    else:
        ferrero_final = {'status':0,'comment':'please provide correct query strings'}
    return JsonResponse(ferrero_final)

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
                if output.get('status',None) == 0:
                    continue
                else:
                    pass
            else:
                output = {'status':0}
            output_sheets[sheet] = output
        output_files[file] = output_sheets
    return JsonResponse(output_files)

@csrf_exempt
def language_detection(request):
    lang , lang_classify = None,None
    # final_lang = {}
    lang_list = set()
    if request.method == 'GET':
        text_list = request.GET.getlist('text',None)
    else:
        text_list = request.POST.getlist('text',None)
    if text_list:
        for text in text_list:
            for sub_text in re.split(r'\s\/',text):
                print('-------'*9)
                print(sub_text)
                print('-------'*9)
                cleaned_text = re.sub(r'\d','',sub_text).lower()
                cleaned_text = re.sub(r'[^\w\s]','',cleaned_text).strip()
                if cleaned_text:
                    if len(cleaned_text.split()) > 5:
                        try:
                            lang = classify(cleaned_text)[0]
                            lang_classify = lang_detect(cleaned_text)
                        except:
                            lang = classify(cleaned_text)[0]
                        finally:
                            if lang and lang_classify:
                                if lang == 'en' or lang_classify == 'en':
                                    lang = 'en'
                                else:
                                    pass
                            else:
                                pass
                    else:
                        try:
                            print('textblob---detection')
                            lang = TextBlob(cleaned_text).detect_language()
                        except:
                            lang = classify(cleaned_text)[0]
                    lang_list.add(lang)
    lang_final = ','.join(list(lang_list))
    return HttpResponse(lang_final)


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
