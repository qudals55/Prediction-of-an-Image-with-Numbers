from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json

# Create your views here.
from django.http import HttpResponse
from django.template import loader 

def index(request):
    template = loader.get_template('ajaxtest/index.html')
    context = {
        'index_test': ['Index Testing Page', 'Is it working?'],
    }
    return HttpResponse(template.render(context, request))

def ajaxform(request):
    template = loader.get_template('ajaxtest/test_form.html')
    context = {'test_message_list': ['hello, there?', 'my name is Hong.', 'How are you, today?'],}  
    return HttpResponse(template.render(context,request))

@csrf_exempt
def searchData(request):
    data = request.POST['msg']
    context = {'msg': data,}
    return HttpResponse(json.dumps(context), "application/json")
