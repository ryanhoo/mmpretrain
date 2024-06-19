const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3000;

const datasetDir = 'datasets/inat-2017'
// 静态文件服务目录
app.use('/dataset', express.static('dataset'));

// API端点，获取图像文件列表
app.get('/api/:task/images', (req, res) => {
  const task = req.params.task;
  const imagesDir = path.resolve(datasetDir, 'images', task);

  fs.readdir(imagesDir, (err, files) => {
    if (err) {
      res.status(500).send('Error reading images directory');
      return;
    }

    const imageFiles = files.filter(file => file.endsWith('.jpg'));
    res.json(imageFiles);
  });
});

// ... (前面的代码保持不变)

// API端点，获取特定图像的标注文件
app.get('/api/:task/labels/:imageName', (req, res) => {
  const task = req.params.task;
  const imageName = req.params.imageName;
  console.log('fetch label by image name:', imageName);
  const labelPath = path.resolve(datasetDir, 'labels', task, imageName.replace('.jpg', '.txt'));
  console.log(labelPath);
  fs.readFile(labelPath, 'utf8', (err, data) => {
    if (err) {
      res.status(500).send('Error reading label file');
      return;
    }
    res.send(data);
  });
});

app.get('/index', (req, res) => {
  const indexPath = path.join(__dirname, 'index.html');
  console.log(indexPath);
  res.sendFile(indexPath);
});

app.get('/:task/images/:imageName', (req, res) => {
  const task = req.params.task;
  const imageName = req.params.imageName;
  const imagePath = path.join(__dirname, '../', datasetDir, 'images', task, imageName);
  console.log(`get image ${imageName} ${datasetDir}`);
  res.sendFile(imagePath);
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});