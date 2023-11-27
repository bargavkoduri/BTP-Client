const fs = require("fs");

const ReadFromFile = (filename) => {
  return new Promise((resolve, reject) => {
    fs.readFile(filename, "utf8", (err, data) => {
      if (err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  });
};

const WriteToFile = (filename, data) => {
  return new Promise((resolve, reject) => {
    fs.writeFile(filename, data, (err) => {
      if (err) reject(err);
      else resolve();
    });
  });
};

module.exports = { ReadFromFile, WriteToFile };
