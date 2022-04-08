
const express = require('express')  
const path = require('path')  
const fs = require('fs');
const archiver = require('archiver');
const crypto = require('crypto')  
const mongoose = require('mongoose')  
const multer = require('multer')  
const {GridFsStorage} = require('multer-gridfs-storage');
const GridFsStream = require('gridfs-stream')  
const bodyParser = require('body-parser')  
const ObjectId = require('mongodb').ObjectId;
  
const app = express()

const mongoURL = 'mongodb://localhost:27017/Federated'  
  
const connect = mongoose.createConnection(mongoURL, {  
    useNewUrlParser: true,  
    useUnifiedTopology: true  
})


let gfs;  
let gridfsBucket;
connect.once('open', () => {  
    // 监听数据库开启，通过 gridfs-stream 中间件和数据库进行文件的出入控制  
    //gfs = GridFsStream(connect.db, mongoose.mongo)  
    // 它会在我们数据库中建立 upload.files(记录文件信息)  upload.chunks(存储文件块)  
    //gfs.collection('upload')  

    gridfsBucket = new mongoose.mongo.GridFSBucket(connect.db, {
        bucketName: 'upload'
    })
    gfs = GridFsStream(connect.db, mongoose.mongo);
    gfs.collection('upload');
})  


const storage = new GridFsStorage({  
    url: mongoURL,  
    file: (req, file) => {  
        return new Promise((resolve, reject) => {  
            // 下面注释部分是给文件进行重命名的，如果想要原文件名称可以自行使用 file.originalname 返回，  
            // crypto.randomBytes(16, (err, buf) => {  
            //     if (err) {  
            //         return reject(err)  
            //     }  
            //     const filename = buf.toString('hex') + path.extname(file.originalname)  
            //     const fileinfo = {  
            //         filename,  
            //         bucketName: 'upload'  
            //     }  
            //     resolve(fileinfo)  
            // })  
            const fileinfo = {  
                //filename: new Date() + '-' + file.originalname,  
                filename: file.originalname,
                bucketName: 'upload'  
            }  
            resolve(fileinfo)  
        })  
    }  
})  
  
const upload = multer({ storage }) 


app.post('/upload', upload.single('model'), (req, res) => {  
    res.send("Upload Local Model Successfully!");
})


app.get('/download', async function(req, res){
    const ids =  [
        ObjectId('624ef778d6a4dda96e0e6d99'),
        ObjectId('624fb50611b491ed1b92b694'), 
    ];
    gfs.files.find({"_id":{"$in":ids}}).toArray((err, files) =>{
        if(!files || files.length === 0) {
            return res.status(404).json({
            err: "No file exist"
            });
        }
        var zipFileName = "aggregation.zip";
        var archive = archiver('zip');
    
        for(var i=0,len=files.length; i<len; i++){
            //console.log(files[i].filename);
            readStream = gridfsBucket.openDownloadStream(files[i]._id);
            archive.append(readStream, {name: files[i].filename});
        }
        archive.finalize();

        res.setHeader('Content-disposition', 'attachment; filename='+zipFileName);
        archive.pipe(res);
      });
    //gfs.files.findOne({_id:ObjectId("624ef778d6a4dda96e0e6d99")}, (err,file)=>{
    // gfs.files.findOne({filename: '0.h5'}, (err,file)=>{ 
    //     if (!file) {
    //         return res.status(404).json({
    //             err: '文件不存在！'
    //         })
    //     }

        // var fn = file.filename 
        // res.set({
        //     //告诉浏览器这是一个二进制文件
        //     "Content-Type": "application/octet-stream",
        //     //告诉浏览器这是一个需要下载的文件，使用encodeURI方法，是为了避免中文名称下载时出问题
        //     "Content-Disposition": `attachment;filename=${encodeURI(fn)}`
        // })
        // const readStream = gridfsBucket.openDownloadStream(file._id);
        // readStream.pipe(res)
    // })
})


app.get('/', async function(request, response) {
    response.sendFile(path.join(__dirname + '/public/upload.html'));
});

var server = app.listen(8080, function () {
    var host = 'localhost'
    var port = server.address().port
    console.log("Example app listening at http://%s:%s", host, port)
});
