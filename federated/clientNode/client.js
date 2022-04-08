'use strict';

var express = require('express');
const FabricCAServices = require('fabric-ca-client');
const { Gateway, Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');
const { SHA3 } = require('sha3');
const { Console } = require('console');
const hash = new SHA3(256);
const prettyPrintJson = require('pretty-print-json');
var bodyParser = require('body-parser');

const MongoClient = require('mongodb').MongoClient;
//const uri = "mongodb+srv://admin:adminpw@blcluster.krh8q.mongodb.net/BL?retryWrites=true&w=majority";
const uri = "mongodb://localhost:27017";
const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });


var Binary = require('mongodb').Binary;
var ObjectId = require('mongodb').ObjectId;

var app = express();


async function main(){
    try {
        await enrollAdmin();
        await registerClient();
    } catch (error) {
        console.error(`Failed!`);
        process.exit(1);
    }

    client.connect(err => {
        var db = client.db("Federated");
        var collection = db.collection("ModelFiles");

        collection.deleteMany({},function(err, res){
            if(err) throw err;
            console.log("Delete old data")
        });
    });
}

async function enrollAdmin(){
    try {
        // load the network configuration
        const ccpPath = path.resolve(__dirname,'..', '..', 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com', 'connection-org1.json');
        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create a new CA client for interacting with the CA.
        const caInfo = ccp.certificateAuthorities['ca.org1.example.com'];
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
            mspId: 'Org1MSP',
            type: 'X.509',
        };
        await wallet.put('admin', x509Identity);
        console.log('Successfully enrolled admin user "admin" and imported it into the wallet');

    } catch (error) {
        console.error(`Failed to enroll admin user "admin": ${error}`);
        process.exit(1);
    }
}

async function registerClient(){
    try {
        // load the network configuration
        const ccpPath = path.resolve(__dirname,'..', '..', 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com', 'connection-org1.json');
        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create a new CA client for interacting with the CA.
        const caURL = ccp.certificateAuthorities['ca.org1.example.com'].url;
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
            mspId: 'Org1MSP',
            type: 'X.509',
        };

        await wallet.put('client', x509Identity);
        console.log('Successfully registered and enrolled admin user "client" and imported it into the wallet');

    } catch (error) {
        console.error(`Failed to register user "client": ${error}`);
        process.exit(1);
    }
}

app.get('/', async function(request, response) {
    response.sendFile(path.join(__dirname + '/public/upload.html'));
});


// 处理上传文件请求
app.post('/upload',async function (request, response) {
 
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
  
    await client.connect(err => {

        var db = client.db("Federated");
        var collection = db.collection("ModelFiles");
        var insertData = {
            org : org,
            fileName : "XXX.pkl",
            fileData : Binary(request.files.avatar.data)
        };
        collection.insertOne(insertData,async function(err){
            if(err) throw err;
            console.log(insertData._id.toString());
            
            try{
                await contract.submitTransaction("UploadLocalModel", org, insertData._id.toString())
            }catch(error){
            
                collection.deleteOne({_id: insertData._id}, function(err){
                    if(err) throw err;
                });
            }
            
        });
    });
 });

 

 var multer = require('multer')

 const storage = multer.diskStorage({

    destination: function (req, file, cb) {
      cb(null, path.resolve(__dirname, './uploads'))
    },
  
    filename: function (req, file, cb) {
      cb(null, file.originalname)
    }
  })
 
 var upload = multer({ storage: storage });
 
 app.post('/test', upload.single('file'), function (req, res) {
 
     console.log(req.body) // form fields
     console.log(req.file) // form files
     res.status(204).end()
 });

 
app.get('/downloadGlobalModel', async function(request,response){
    console.log("download global model");

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
         collection.findOne(({_id: ObjectId(globalModel.curGlobalModelId)}),function(err,res){
            fs.writeFile(path.resolve(__dirname,"global.txt"),res.fileData.buffer,function(err){
                if(err) throw err;
            });
        })
    }
    else {
        console.log("Global Model is empty");
    }
});

main();

var server = app.listen(8080, function () {
    var host = 'localhost'
    var port = server.address().port
    console.log("Example app listening at http://%s:%s", host, port)
});
