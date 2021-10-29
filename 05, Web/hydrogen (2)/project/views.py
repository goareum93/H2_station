from django.shortcuts import render

# Create your views here.
def chart(request):
    return render(request,'project/chart.html')

def empty(request):
    return render(request,'project/empty.html')

def form(request):
    return render(request,'project/form.html')

def index(request):
    return render(request,'project/index.html')

def tab_panel(request):
    return render(request,'project/tab-panel.html')

def table(request):
    return render(request,'project/table.html')

def ui_elements(request):
    return render(request,'project/ui-elements.html')