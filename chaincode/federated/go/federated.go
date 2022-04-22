package main

import (
	"encoding/json"
	"encoding/pem"
    "crypto/x509"
	"strconv"
	"fmt"
    "bytes"
	//"sync/atomic"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"

    "gopkg.in/mgo.v2"
    "gopkg.in/mgo.v2/bson"

	"os"
    "io"
	"path/filepath"
)

// SmartContract provides functions for managing a car
type SmartContract struct {
	contractapi.Contract
}

type OrgInfo struct {
	CurModelId string `json:"curModelId"`
	CurWeight float32 `json:"curWeight"`
	//Org string `json:"org"`
	//Cert string `json:"cert"`
	//Signature
}

type ModelBlock struct {
	ModelType string `json:"modelType"`
	PrevModelId string`json:"prevModelId"`
	Weight float32 `json:"weight"`
	Org string `json:"org"`
	//WeightHash uint32 `json:"weightHash"`
	//Timestamp int64 `json:"timestamp"`
	//VerifyCode uint32 `json:verifyCode""`
}

type GlobalModel struct {
	CurGlobalModelId string `json:"curGlobalModelId"`
    Round uint32 `json:"round"`
	LocalModelIds []string `json:"localModelIds"`
	UploadCount uint32 `json:"uploadCount"`
	TriggerNum uint32 `json:"triggerNum"`
}

//-------------------------------------------

type AggregateInfo struct {
	GlobalModel *GlobalModel `json:"globalModel"`
	ModelBlocks []ModelBlock `json:"modelBlocks"`
}

var session *mgo.Session

type Test struct {
	Name string `bson:"name"`
}


// InitLedger adds a base set of cars to the ledger
func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {

	var globalModel GlobalModel = GlobalModel{CurGlobalModelId:"" ,Round:0, LocalModelIds:[]string{}, UploadCount:0, TriggerNum:3}
	globalModelBytes, _ := json.Marshal(globalModel)
	err := ctx.GetStub().PutState("global",globalModelBytes)

	if err != nil {
		return fmt.Errorf("Failed to put to world state. %s", err.Error())
	}

	for i := 1; i <= 10; i++ {
		var orgInfo OrgInfo = OrgInfo{CurModelId:"-", CurWeight:1.0}
		orgInfoByte, _ := json.Marshal(orgInfo)
		err = ctx.GetStub().PutState("org"+strconv.Itoa(i), orgInfoByte)
	}

	se, _ := mgo.Dial("114.212.82.53:27017")
	session = se
	return nil
}

func Connect(db, collection string) (*mgo.Session, *mgo.Collection) {
	ms := session.Copy()
	c := ms.DB(db).C(collection)
	ms.SetMode(mgo.Monotonic, true)
	return ms, c
}

func Insert(db, collection string, doc interface{}) error {
	ms, c := Connect(db, collection)
	defer ms.Close()

	return c.Insert(doc)
}

func FindOne(db, collection string, query, result interface{}) error {
	ms, c := Connect(db, collection)
	defer ms.Close()

	return c.Find(query).One(result)
}

func LoadFile(filename string) error{
	ms := session.Copy()
	defer ms.Close()
	db := ms.DB("Federated")
	file, err := db.GridFS("file").Open(filename)
	if err != nil {
        return err
    }
    defer file.Close()

	out, err := os.OpenFile("./1.txt", os.O_CREATE, 0666)
	if err != nil {
		return err
	}
    _,err = io.Copy(out, file)

	return nil
}


func (s *SmartContract) TestMongo(ctx contractapi.TransactionContextInterface) ([]string,error){
	// session, err := mgo.Dial("114.212.82.53:27017")
    // if err != nil {
    //     panic(err)
    // }
	// s.session = session
	// defer s.session.Close()

    // s.session.SetMode(mgo.Monotonic, true)
    // db := s.session.DB("Federated")
	// c := db.C("Test")


	// test := Test{}
	// c.Find(bson.M{"name": "abcd"}).One(&test)
	// return &test,nil
	os.Mkdir("./benben",0777)
	os.Mkdir("/benben",0777)
    test := &Test{ Name:"whh" } 
	Insert("Federated","Test", test)

	var result Test
	FindOne("Federated","Test", bson.M{"name": "abcd"}, &result)

	LoadFile("1.txt")

	str, _ := os.Getwd()
	filepathNames, _ := filepath.Glob(filepath.Join(str, "*"))
	return filepathNames, nil 
}
 
func (s *SmartContract) GetOrg(ctx contractapi.TransactionContextInterface) (string,error){

    creatorByte,_ := ctx.GetStub().GetCreator()
    certStart := bytes.IndexAny(creatorByte,"-----BEGIN")
    if(certStart==-1){
        return "",fmt.Errorf("No Cert Found")
    }
    certText := creatorByte[certStart:]
    bl,_ := pem.Decode(certText)
    if (bl==nil){
        return "",fmt.Errorf("Decode PEM Failed")
    }
    cert,err := x509.ParseCertificate(bl.Bytes)
    if (err!=nil){
        return "",fmt.Errorf("Parse Cert Failed")
    }
    uname := cert.Subject.CommonName
    //return "",fmt.Errorf(uname)
    return uname,nil

}

func (s *SmartContract) GetOrgInfo(ctx contractapi.TransactionContextInterface,org string) (*OrgInfo,error) {
	orgInfoBytes, err := ctx.GetStub().GetState(org)
	if err != nil {
		return nil, fmt.Errorf("%s",err.Error())
    }

    var  orgInfo OrgInfo
    _ = json.Unmarshal(orgInfoBytes, &orgInfo)
    return &orgInfo,err
}

func (s *SmartContract) GetGlobalModel(ctx contractapi.TransactionContextInterface) (*GlobalModel,error) {
   globalModelBytes, err := ctx.GetStub().GetState("global")
   if err != nil {
		return nil, fmt.Errorf("%s",err.Error())
   }

   var globalModel GlobalModel
   _ = json.Unmarshal(globalModelBytes, &globalModel)
   return &globalModel,err
}

func (s *SmartContract) GetModelBlock(ctx contractapi.TransactionContextInterface, curModelId string) (*ModelBlock,error) {
	modelBlockBytes, err := ctx.GetStub().GetState(curModelId);
	if err != nil {
		return nil, fmt.Errorf("%s",err.Error())
	}

	var modelBlock ModelBlock
	_ = json.Unmarshal(modelBlockBytes, &modelBlock)
	return &modelBlock,err
}

func (s *SmartContract) DownloadGlobalModelInfo(ctx contractapi.TransactionContextInterface) (*AggregateInfo, error){
	var info AggregateInfo
	
	globalModel, err := s.GetGlobalModel(ctx)
	info.GlobalModel = globalModel

	blocks := []ModelBlock{}
	for i := 0; i < len(globalModel.LocalModelIds); i++ {
		
		modelBlock, _ := s.GetModelBlock(ctx, globalModel.LocalModelIds[i])
		//fmt.Printf(modelBlock.ModelType)
		blocks = append(blocks, *modelBlock)
	}   
	info.ModelBlocks = blocks

    return &info, err
}



func (s *SmartContract) UpdateLocalModel(ctx contractapi.TransactionContextInterface, org string, curModelId string) error{
	orgInfo, err := s.GetOrgInfo(ctx,org)
	if err != nil {
		return fmt.Errorf("%s",err.Error())
	}

	var modelBlock ModelBlock = ModelBlock{ModelType:"Local", PrevModelId:orgInfo.CurModelId, Weight:orgInfo.CurWeight, Org:org}
	orgInfo.CurModelId = curModelId

	modelBlockBytes, _ := json.Marshal(modelBlock)
	err = ctx.GetStub().PutState(curModelId, modelBlockBytes)

	orgInfoBytes, _ := json.Marshal(orgInfo)
	err = ctx.GetStub().PutState(org,orgInfoBytes)

	if err != nil {
		return fmt.Errorf("Failed to put to world state. %s", err.Error())
	}

	//s.UpdateGlobalModel(ctx,curModelId)
	globalModel, err :=s.GetGlobalModel(ctx);
	globalModel.UploadCount +=1;
	globalModel.LocalModelIds = append(globalModel.LocalModelIds, curModelId)

	globalModelBytes, _ := json.Marshal(globalModel)
	err = ctx.GetStub().PutState("global",globalModelBytes)

	if err != nil {
		return fmt.Errorf("%s",err.Error())
    }
 
	if globalModel.UploadCount == globalModel.TriggerNum {
		fmt.Printf("Begin Global Model Federated Aggregation")
	}
	return nil
}

func (s *SmartContract) FedAvg(ctx contractapi.TransactionContextInterface, weights string) error{
	//something to do

	weightsMap := make(map[string]float32)
	err := json.Unmarshal([]byte(weights), &weightsMap)
	
	for k,v := range weightsMap{
		orgInfo, _ := s.GetOrgInfo(ctx,k)
		orgInfo.CurWeight = v
		orgInfoBytes, _ := json.Marshal(orgInfo)
		err = ctx.GetStub().PutState(k,orgInfoBytes)
	}
	// orgInfoBytes, _ := json.Marshal(orgInfo)
	// err = ctx.GetStub().PutState("org1",orgInfoBytes)
	return err
	// globalModel, err :=s.GetGlobalModel(ctx);
	// globalModel.UploadCount +=1;
	// globalModel.LocalModelIds = append(globalModel.LocalModelIds, localModelId)

	// globalModelBytes, _ := json.Marshal(globalModel)
	// err = ctx.GetStub().PutState("global",globalModelBytes)

	// if err != nil {
	// 	return fmt.Errorf("%s",err.Error())
    // }
 
	// if globalModel.UploadCount == globalModel.TriggerNum {
	// 	fmt.Printf("Begin Global Model Federated Aggregation")
	// }
	return nil
}


func main() {

	chaincode, err := contractapi.NewChaincode(new(SmartContract))

	if err != nil {
		fmt.Printf("Error create fabcar chaincode: %s", err.Error())
		return
	}

	if err := chaincode.Start(); err != nil {
		fmt.Printf("Error starting fabcar chaincode: %s", err.Error())
	}
}