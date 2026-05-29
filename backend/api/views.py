from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .ml_utils import run_inference_and_xai

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']
        try:
            image_bytes = image_file.read()
            results = run_inference_and_xai(image_bytes)
            return Response(results, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
