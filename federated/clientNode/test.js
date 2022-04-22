
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

const FabricCAServices = require('fabric-ca-client');
const { Gateway, Wallets } = require('fabric-network');

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


app.post('/upload', upload.single('model'), async function(req, res){  

    const ccpPath = path.resolve(__dirname,'..', '..','test-network', 'organizations', 'peerOrganizations', 'org1.example.com', 'connection-org1.json');
    let ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

    // Create a new file system based wallet for managing identities.
    const walletPath = path.join(process.cwd(), 'wallet');
    const wallet = await Wallets.newFileSystemWallet(walletPath);
    console.log(`Wallet path: ${walletPath}`);

    // Check to see if we've already enrolled the user.
    const identity = await wallet.get('client');
    if (!identity) {
        console.log('An identity for the user "client" does not exist in the wallet');
        return;
    }

    // Create a new gateway for connecting to our peer node.
    const gateway = new Gateway();
    await gateway.connect(ccp, { wallet, identity: 'client', discovery: { enabled: true, asLocalhost: true } });

    // Get the network (channel) our contract is deployed to.
    const network = await gateway.getNetwork('mychannel');

    // Get the contract from the network.
    const contract = network.getContract('federated');

    //var org = await contract.evaluateTransaction("GetOrg");
    //org = org.match(/org[0-9]+/)[0];
    //console.log(org);
    var org = "org1";

    gfs.files.findOne({"filename": req.file.originalname},async function(err, file){
       console.log(file._id.toString());

       try{
            await contract.submitTransaction("UpdateLocalModel", org, file._id.toString())
       }catch(error){
            //collection.deleteOne({_id: insertData._id}, function(err){
            if(err) throw err;
       };
    });


    res.send("Upload Local Model Successfully!");
})


app.get('/downloadGlobal', async function(req, res){

    const ccpPath = path.resolve(__dirname,'..', '..','test-network', 'organizations', 'peerOrganizations', 'org1.example.com', 'connection-org1.json');
    let ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

    // Create a new file system based wallet for managing identities.
    const walletPath = path.join(process.cwd(), 'wallet');
    const wallet = await Wallets.newFileSystemWallet(walletPath);
    console.log(`Wallet path: ${walletPath}`);

    // Check to see if we've already enrolled the user.
    const identity = await wallet.get('client');
    if (!identity) {
        console.log('An identity for the user "client" does not exist in the wallet');
        return;
    }

    // Create a new gateway for connecting to our peer node.
    const gateway = new Gateway();
    await gateway.connect(ccp, { wallet, identity: 'client', discovery: { enabled: true, asLocalhost: true } });

    // Get the network (channel) our contract is deployed to.
    const network = await gateway.getNetwork('mychannel');

    // Get the contract from the network.
    const contract = network.getContract('federated');

    var globalModel = await contract.evaluateTransaction("GetGlobalModel");
    var globalModel = JSON.parse(globalModel.toString());

    if(globalModel.curGlobalModelId != ""){
        gfs.files.findOne({"_id": ObjectId(globalModel.curGlobalModelId)},async function(err, file){
            console.log(file);
            if(!files || files.length === 0) {
                return res.status(404).json({
                err: "No file exist"
                });
            }
            var zipFileName = "global.zip";
            var archive = archiver('zip');
        
            readStream = gridfsBucket.openDownloadStream(file._id);
            archive.append(readStream, {name: file.filename});
            archive.finalize();
    
            res.setHeader('Content-disposition', 'attachment; filename='+zipFileName);
            archive.pipe(res);
        });
    }


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

app.get('/downloadLocals', async function(req, res){
    const ccpPath = path.resolve(__dirname,'..', '..','test-network', 'organizations', 'peerOrganizations', 'org1.example.com', 'connection-org1.json');
    let ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

    // Create a new file system based wallet for managing identities.
    const walletPath = path.join(process.cwd(), 'wallet');
    const wallet = await Wallets.newFileSystemWallet(walletPath);
    console.log(`Wallet path: ${walletPath}`);

    // Check to see if we've already enrolled the user.
    const identity = await wallet.get('client');
    if (!identity) {
        console.log('An identity for the user "client" does not exist in the wallet');
        return;
    }

    // Create a new gateway for connecting to our peer node.
    const gateway = new Gateway();
    await gateway.connect(ccp, { wallet, identity: 'client', discovery: { enabled: true, asLocalhost: true } });

    // Get the network (channel) our contract is deployed to.
    const network = await gateway.getNetwork('mychannel');

    // Get the contract from the network.
    const contract = network.getContract('federated');

    var globalModel = await contract.evaluateTransaction("GetGlobalModel");
    var globalModel = JSON.parse(globalModel.toString());

    var ids = globalModel.localModelIds;
    for(var i=0;i<ids.length;i++) ids[i] = ObjectId(ids[i]);
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
});

app.get('/test', async function(req, res){
    const ccpPath = path.resolve(__dirname,'..', '..','test-network', 'organizations', 'peerOrganizations', 'org1.example.com', 'connection-org1.json');
    let ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

    // Create a new file system based wallet for managing identities.
    const walletPath = path.join(process.cwd(), 'wallet');
    const wallet = await Wallets.newFileSystemWallet(walletPath);
    console.log(`Wallet path: ${walletPath}`);

    // Check to see if we've already enrolled the user.
    const identity = await wallet.get('client');
    if (!identity) {
        console.log('An identity for the user "client" does not exist in the wallet');
        return;
    }

    // Create a new gateway for connecting to our peer node.
    const gateway = new Gateway();
    await gateway.connect(ccp, { wallet, identity: 'client', discovery: { enabled: true, asLocalhost: true } });

    // Get the network (channel) our contract is deployed to.
    const network = await gateway.getNetwork('mychannel');

    // Get the contract from the network.
    const contract = network.getContract('federated');

    //var globalModel = await contract.evaluateTransaction("DownloadGlobalModelInfo");
    //console.log(globalModel.toString());

    var m = {};
    m["org1"] = 0.1;
    m["org2"] = 0.3;
    m["org4"] = 0.15;
    await contract.submitTransaction("FedAvg",JSON.stringify(m));
    
});

app.get('/', async function(request, response) {
    response.sendFile(path.join(__dirname + '/public/upload.html'));
});

var server = app.listen(8080, function () {
    var host = 'localhost'
    var port = server.address().port
    console.log("Example app listening at http://%s:%s", host, port)
});


async function enrollAdmin(orgId){
    try {
        // load the network configuration
        const ccpPath = path.resolve(__dirname,'..', '..', 'test-network', 'organizations', 'peerOrganizations', 'org'+orgId+'.example.com', 'connection-org'+orgId+'.json');
        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create a new CA client for interacting with the CA.
        const caInfo = ccp.certificateAuthorities['ca.org'+orgId+'.example.com'];
        const caTLSCACerts = caInfo.tlsCACerts.pem;
        const ca = new FabricCAServices(caInfo.url, { trustedRoots: caTLSCACerts, verify: false }, caInfo.caName);

        // Create a new file system based wallet for managing identities.
        const walletPath = path.join(process.cwd(), 'wallet');
        const wallet = await Wallets.newFileSystemWallet(walletPath);
        console.log(`Wallet path: ${walletPath}`);

        // Check to see if we've already enrolled the admin user.
        const identity = await wallet.get('admin');
        if (identity) {
            console.log('An identity for the admin user "admin" already exists in the wallet');
            return;
        }

        // Enroll the admin user, and import the new identity into the wallet.
        const enrollment = await ca.enroll({ enrollmentID: 'admin', enrollmentSecret: 'adminpw' });
        const x509Identity = {
            credentials: {
                certificate: enrollment.certificate,
                privateKey: enrollment.key.toBytes(),
            },
            mspId: 'Org'+orgId+'MSP',
            type: 'X.509',
        };
        await wallet.put('admin', x509Identity);
        console.log('Successfully enrolled admin user "admin" and imported it into the wallet');

    } catch (error) {
        console.error(`Failed to enroll admin user "admin": ${error}`);
        process.exit(1);
    }
}

async function registerClient(orgId){
    try {
        // load the network configuration
        //const ccpPath = path.resolve(__dirname,'..', '..', 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com', 'connection-org1.json');
        const ccpPath = path.resolve(__dirname,'..', '..', 'test-network', 'organizations', 'peerOrganizations', 'org'+orgId+'.example.com', 'connection-org'+orgId+'.json');
        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create a new CA client for interacting with the CA.
        //const caURL = ccp.certificateAuthorities['ca.org1.example.com'].url;
        const caURL = ccp.certificateAuthorities['ca.org'+orgId+'.example.com'];
        const ca = new FabricCAServices(caURL);

        // Create a new file system based wallet for managing identities.
        const walletPath = path.join(process.cwd(), 'wallet');
        const wallet = await Wallets.newFileSystemWallet(walletPath);
        console.log(`Wallet path: ${walletPath}`);

        // Check to see if we've already enrolled the user.
        const userIdentity = await wallet.get('client');
        if (userIdentity) {
            console.log('An identity for the user "client" already exists in the wallet');
            return;
        }

        // Check to see if we've already enrolled the admin user.
        const adminIdentity = await wallet.get('admin');
        if (!adminIdentity) {
            console.log('An identity for the admin user "admin" does not exist in the wallet');
            return;
        }

        // build a user object for authenticating with the CA
        const provider = wallet.getProviderRegistry().getProvider(adminIdentity.type);
        const adminUser = await provider.getUserContext(adminIdentity, 'admin');

        // Register the user, enroll the user, and import the new identity into the wallet.
        const secret = await ca.register({
            affiliation: 'org1.department1',
            enrollmentID: 'client',
            role: 'client'
        }, adminUser);
        const enrollment = await ca.enroll({
            enrollmentID: 'client',
            enrollmentSecret: secret
        });
        const x509Identity = {
            credentials: {
                certificate: enrollment.certificate,
                privateKey: enrollment.key.toBytes(),
            },
            mspId: 'Org'+orgId+'MSP',
            type: 'X.509',
        };

        await wallet.put('client', x509Identity);
        console.log('Successfully registered and enrolled admin user "client" and imported it into the wallet');

    } catch (error) {
        console.error(`Failed to register user "client": ${error}`);
        process.exit(1);
    }
}

async function deleteGFS(){

    let allFiles = await gfs.files.find({}).toArray();
    for (let file of allFiles) {
        await gridfsBucket.delete(file._id);
    }
    console.log("Delete all old files");
}

async function main(){
    try {
        await enrollAdmin(1);
        await registerClient(1);
    } catch (error) {
        console.error(`Failed!`);
        process.exit(1);
    }

    deleteGFS();
}

main();