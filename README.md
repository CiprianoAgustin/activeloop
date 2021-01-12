### What is Hub for?

Software 2.0 needs Data 2.0, and Hub delivers it. Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of training models. With Hub, we are fixing this. We store your (even petabyte-scale) datasets as single numpy-like array on the cloud, so you can seamlessly access and work with it from any machine. Hub makes any data type (images, text files, audio, or video) stored in cloud usable as fast as if it were stored on premise. With same dataset view, your team can always be in sync. 

### Features 

* Store and retrieve large datasets with version-control
* Collaborate as in Google Docs: Multiple data scientists working on the same data in sync with no interruptions
* Access from multiple machines simultaneously
* Integrate with your ML tools like Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), or [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* Deploy on Google Cloud, S3, Azure as well as Activeloop (by default - and for free!) 
* Create arrays as big as you want. You can store images as big as 100k by 100k!
* Keep shape of each sample dynamic. This way you can store small and big arrays as 1 array. 
* [Visualize](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) any slice of the data in a matter of seconds without redundant manipulations

## Getting Started

### Access public data. Fast

To load a public dataset, one needs to write dozens of lines of code and spend hours accessing and understanding the API, as well as downloading the data. With Hub, you only need 2 lines of code, and you **can get started working on your dataset in under 3 minutes**. 

```sh
pip3 install hub
```

You can access public datasets with a few lines of code.
```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")
mnist["image"][0:1000].compute()
```

### Train a model

Load the data and directly train your model using pytorch

```python
from hub import Dataset
import torch

mnist = Dataset("activeloop/mnist")
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Training loop here
```

### Create a local dataset 

```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "./data/dataset_name",
    shape = (4,),
    mode = "w+",
    schema = {
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.commit()
```

You can also specify `s3://bucket/path`, `gcs://bucket/path` or azure path [more here](https://docs.activeloop.ai/en/latest/simple.html#data-storage).

### Upload your dataset and access it from <ins>anywhere</ins> in 3 simple steps

1. Register a free account at [Activeloop](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) and authenticate locally
```sh
hub register
hub login
```

2. Then create a dataset and upload
```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "username/dataset_name",
    shape = (4,),
    mode = "w+",
    schema = {
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.commit()
```

3. Access it from anywhere else in the world, on any device having a command line.
```python
from hub import Dataset

ds = Dataset("username/dataset_name")
```


## Documentation

For more advanced data pipelines like uploading large datasets or applying many transformations, please read the [docs](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

## Use Cases
* **Satellite and drone imagery**: [Smarter farming with scalable aerial pipelines](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Mapping Economic Well-being in India](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Fighting desert Locust in Kenya with Red Cross](https://omdena.com/projects/ai-desert-locust/)
* **Medical Images**: Volumetric images such as MRI or Xray
* **Self-Driving Cars**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Retail**: Self-checkout datasets
* **Media**: Images, Video, Audio storage
