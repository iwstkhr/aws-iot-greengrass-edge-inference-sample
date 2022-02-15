from sagemaker.pytorch import PyTorch


AWS_ACCOUNT_ID = '123456789012'
# An S3 bucket used by SageMaker
S3_BUCKET = f's3://sagemaker-ml-model-artifacts-{AWS_ACCOUNT_ID}-ap-northeast-1'


if __name__ == '__main__':
    # Specify a training job spec.
    # see https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html
    # see https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
    pytorch_estimator = PyTorch(
        entry_point='training.py',
        source_dir='./',
        role='sagemaker-execution-role',
        instance_count=1,
        instance_type='ml.m5.large',
        framework_version='1.10.0',
        py_version='py38',
        output_path=f'{S3_BUCKET}/models/trained',
        hyperparameters={}
    )
    # Call `fit` with your S3 bucket to create a training job and start training.
    # In this example, there is no actual training code, so call `fit` with no channel arguments.
    # see https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase.fit
    # pytorch_estimator.fit({'train': f'{S3_BUCKET}/path/to/train/'})
    pytorch_estimator.fit()
