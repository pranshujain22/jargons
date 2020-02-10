from django.http import HttpResponseRedirect, JsonResponse
from django.core.files.storage import FileSystemStorage
from django.urls.base import reverse
from django.contrib import messages
from django.shortcuts import render
import dashboard.Elastic as elastic
from django.views.decorators.csrf import csrf_exempt
import dashboard.prediction as prediction


def index(request):
    message = None
    for msg in messages.get_messages(request):
        message = msg
        break
    return render(request, 'dashboard/home.html', {'message': message})


def upload(request):
    if request.method == "POST":
        try:
            url = "/api/v2/containers/"
            context = {}
            uploaded_file = request.FILES['csv-file']
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            print(name)
            context['url'] = fs.url(name)

            elastic.csv_writer(fs.path(name), str(name).split('.')[0].lower(), 'data', ';')
            fs.delete(name)
            messages.add_message(request, messages.INFO, 'File uploaded successfully')
            return HttpResponseRedirect(reverse('dashboard:index'))
        except Exception as e:
            print(e)
            messages.add_message(request, messages.INFO, 'Failed to upload file...!')
            return HttpResponseRedirect(reverse('dashboard:index'))

    message = None
    for msg in messages.get_messages(request):
        message = msg
        break
    return render(request, 'dashboard/home.html', {'message': message})


def indices(request):
    if request.method == 'GET':
        indices = list(elastic.get_indices())
        print('indices', indices)
        return JsonResponse({'indices': indices})

    return JsonResponse({})


@csrf_exempt
def get_prediction(request):
    if request.method == 'POST':
        index_name = request.POST.get('index')
        print(index_name)
        data = dict()
        predictions, test = prediction.predict(index_name)

        y_predict = list()
        for p in predictions:
            y_predict.append(p.tolist()[0])

        y_test = list()
        for t in test:
            y_test.append(t.tolist()[0])

        data['prediction'] = {
            'x': list(range(len(y_predict))),
            'y': y_predict
        }
        data['test'] = {
            'x': list(range(len(y_test))),
            'y': y_test
        }
        print(data)
        return JsonResponse(dict(data))

    return JsonResponse({})

