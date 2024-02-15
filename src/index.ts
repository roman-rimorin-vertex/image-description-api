import express from 'express';
import fs from 'fs';
// Import Multer for handling multipart/form-data, used for uploading files.
import multer from 'multer';
// Import TensorFlow.js for Node.js to utilize machine learning models.
import * as tf from '@tensorflow/tfjs-node';
// Import the MobileNet model, pre-trained for image classification.
import * as mobilenet from '@tensorflow-models/mobilenet';

const upload = multer({ dest: 'uploads/' });

const app = express();
const port = 3000;

// Define a POST route for '/describe-image' to handle image uploads.
app.post('/describe-image', upload.single('image'), async (req, res) => {
    // Check if an image file was uploaded. If not, return a 400 error.
    if (!req.file)
        return res.status(400).send('No image uploaded.');

    // Read the uploaded image file asynchronously.
    const image = await fs.promises.readFile(req.file.path);
    // Decode the image file into a tensor with 3 color channels.
    let decodedImage = tf.node.decodeImage(image, 3);
    // Load the MobileNet model asynchronously.
    const model = await mobilenet.load();
    
    // Initialize an array to hold the classification predictions.
    let predictions: {
        className: string;
        probability: number;
    }[];

    // Check if the tensor is 4D (indicating a batch of images). If so, adjust to 3D.
    if (decodedImage.rank === 4) 
        // Classify the image after adjusting the tensor's shape, and store predictions.
        predictions = await model.classify(decodedImage.squeeze([0]) as tf.Tensor3D);
    else 
        // Directly classify the image as its tensor is already 3D, and store predictions.
        predictions = await model.classify(decodedImage as tf.Tensor3D);

    // Send the classification predictions as a JSON response.
    res.json(predictions);

    // Delete the uploaded image file to clean up the server storage.
    await fs.promises.unlink(req.file.path);
});

// Start the server and listen on the specified port, log a message upon starting.
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
