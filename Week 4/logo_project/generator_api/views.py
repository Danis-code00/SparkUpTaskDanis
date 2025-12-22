import shutil
from pathlib import Path
from django.shortcuts import render
from django.http import FileResponse
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from .tasks import generate_logo_task  # Fixed: removed the stray quote

# 1. The Dashboard View
def home_view(request):
    return render(request, 'index.html')

# 2. The Generation API
class GenerateView(APIView):
    def post(self, request):
        count_raw = request.data.get('count', 10)
        try:
            count = int(count_raw) 
        except (ValueError, TypeError):
            count = 10 
            
        generate_logo_task(count)
        return Response({"status": f"Started generating {count} images!"})

# 3. The Download API
class DownloadDatasetView(APIView):
    def get(self, request):
        dataset_path = Path(settings.BASE_DIR) / "dataset"
        zip_name = Path(settings.BASE_DIR) / "logo_dataset"
        
        if not dataset_path.exists():
            return Response({"error": "No dataset found. Generate some images first!"}, status=404)

        # ZIP the dataset folder
        zip_file_path = shutil.make_archive(str(zip_name), 'zip', str(dataset_path))
        
        return FileResponse(open(zip_file_path, 'rb'), as_attachment=True, filename="logo_dataset.zip")