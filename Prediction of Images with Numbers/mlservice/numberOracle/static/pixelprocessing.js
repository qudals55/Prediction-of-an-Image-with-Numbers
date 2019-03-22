var img = new Image();

//function load(img) {
img.src = 't10k-0-7.jpg';
img.onload = function() {
  draw(this);
};
//};

function draw(img) {
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  img.style.display = 'none';
  var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  var data = imageData.data;
/*
  var load = function() {
    ctx.drawImage(img, 0, 0);
    img.style.display = 'none';
    img.src = gname; //'t10k-0-7.jpg';
    imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    data = imageData.data;
  };
*/
  var invert = function() {
    for (var i = 0; i < data.length; i += 4) {
      data[i]     = 255 - data[i];     // red
      data[i + 1] = 255 - data[i + 1]; // green
      data[i + 2] = 255 - data[i + 2]; // blue
    }
    ctx.putImageData(imageData, 0, 0);
  };

  var grayscale = function() {
    for (var i = 0; i < data.length; i += 4) {
      var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i]     = avg; // red
      data[i + 1] = avg; // green
      data[i + 2] = avg; // blue
    }
    ctx.putImageData(imageData, 0, 0);
  };

  var random = function() {
       for (var i=0; i < 30000; i += 4) {
           var p = Math.floor(Math.random() * data.length / 4) * 4;
           data[p] = Math.floor(Math.random() * 256);
           data[p+1] = Math.floor(Math.random() * 256);
           data[p+2] = Math.floor(Math.random() * 256);
           data[p+3] = 255;
       }
       ctx.putImageData(imageData, 0, 0);
  }

  var loadbtn = document.getElementById('loadbtn');
  loadbtn.onclick = function(){load(img);};
  var invertbtn = document.getElementById('invertbtn');
  invertbtn.addEventListener('click', invert);
  var grayscalebtn = document.getElementById('grayscalebtn');
  grayscalebtn.addEventListener('click', grayscale);
  var randombtn = document.getElementById('randombtn');
  randombtn.addEventListener('click', random);
}
