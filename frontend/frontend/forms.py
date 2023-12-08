from django import forms

class Query(forms.Form):
    query=forms.CharField(required=True)