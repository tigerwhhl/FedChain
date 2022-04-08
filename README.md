# FedChain
Blockchain for Federated Machine Learning

## Default Network Configuration:

The default application uses the HyperLedger Test network which bootstraps the following instances:

1. 1 Orderer
2. 1 Certifying Authority
3. 2 org (org0 and org1) maintaining 2 peer (peer0 and peer 1)
4. 1 CouchDB
5. 1 CLI

## Usage Instructions:

### Prerequisites:

1. [HyperLedger Fabric v2.2.x LTS](https://www.hyperledger.org/projects/fabric "HyperLedger Fabric Homepage")
2. Download this repository, and merge BEAS/fabric-samples with the HyperLedger fabric-samples directory.

### Running the Application:

#### Network Setup:

```
$  cd fabric-samples/federated
$  ./end.sh
$  ./start.sh
```
#### Create a client node for uploading and downloading the models

1. Change working directory:
```
$  cd fabric-samples/federated/clientNode
```
2. Install Application Dependancies:
```
$  npm install
```
3. Run Application
```
$  node client.js
```

4. To view the frontend, go to your browser and lauch http://localhost:8080
