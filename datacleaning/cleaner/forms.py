from django import forms
   
# creating a form 
class InputForm(forms.Form):
    colName = forms.CharField(max_length = 200, required = False)