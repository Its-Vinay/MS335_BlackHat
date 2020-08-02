from django import forms

class BmiInputForm(forms.Form):
    input_img = forms.ImageField(required=True)