const express = require('express');
const fileUpload = require('express-fileupload');
const app = express();

// default options
app.use(fileUpload());
 
app.post('/upload', function(req, res) {
  if (!req.files)
    return res.status(400).send('No files were uploaded.');
 
  // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
  let sampleFile = req.files.sampleFile;
 
  // Use the mv() method to place the file somewhere on your server
  sampleFile.mv('upload.jpg', function(err) {
    if (err) return res.status(500).send(err);
    
    var request = require('request');
    request('http://192.168.0.11:5000/predict', function (error, response, body) {
        console.log('error:', error); // Print the error if one occurred
        console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
        console.log('body:', body); // Print the HTML for the Google homepage.
      res.send(body)
    });
    
  });
});

app.listen(8000, () => console.log('Example app listening on port 8000!'))