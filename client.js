let io = require("socket.io-client");
const { PythonShell } = require("python-shell");
const { WriteToFile } = require("./helper");
const ss = require("socket.io-stream");
const fs = require("fs");

// connect to server
let socket = io.connect("http://20.193.139.159:5000", {
  reconnection: true,
  maxHttpBufferSize: 3e9,
});

socket.on("connect", () => {
  // receive global model
  socket.on("global_model_config", (data) => {
    // saving the global model
    try {
      WriteToFile("model_config.json", JSON.stringify(data.model_config));
      console.log("Received Model config...");
      console.log("Waiting to start training...\n");
    } catch (err) {
      console.log(err);
    }
  });

  // receive weights for training
  ss(socket).on("start_train", (stream) => {
    console.log("\nSelected for Training");
    stream.pipe(fs.createWriteStream("model_weights.txt"));
    stream.on("end", () => {
      console.log("starting training...");
      PythonShell.run("client.py", { scriptPath: "" }, (err) => {
        console.log("Training Complete");
        if (!err) {
          let stream = ss.createStream();
          ss(socket).emit("weight_update", stream);
          fs.createReadStream("model_weights_updated.txt").pipe(stream);
          stream.on("end", () => {
            console.log("Weights Updates sent");
          });
        } else console.log(err);
      });
    });
  });

  socket.on("disconnect", function (error) {
    console.log("\nMessage :", error);
  });
});
