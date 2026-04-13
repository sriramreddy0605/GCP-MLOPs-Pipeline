import kfp
from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform

@dsl.component(packages_to_install=['scikit-learn', 'joblib', 'pandas', 'google-cloud-storage'])
def train_iris_model(project_id: str, bucket_name: str):
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    from google.cloud import storage
    import os

    # 1. Train
    iris = load_iris()
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)

    # 2. Save and Upload
    model_file = 'model.joblib'
    joblib.dump(model, model_file)
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f'models/{model_file}')
    blob.upload_from_filename(model_file)
    print(f"Model stored in gs://{bucket_name}/models/{model_file}")

@dsl.pipeline(name='mlops-iris-pipeline')
def mlops_pipeline(project_id: str, bucket_name: str):
    train_task = train_iris_model(project_id=project_id, bucket_name=bucket_name)

if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=mlops_pipeline, package_path='pipeline.json')
