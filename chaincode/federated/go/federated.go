package main

import (
	"encoding/json"
	"strconv"
	"fmt"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"time"
)

// SmartContract provides functions for managing a car
type SmartContract struct {
	contractapi.Contract
}

// model detail information
// k: curHashId v:ModelBlock
type ModelBlock struct {
	ModelType string `json:"modelType"` 
	PrevHashId string `json:"prevHashId"` 
	ModelUrl string `json:"modelUrl"`
	Timestamp string `json:"timestamp"`
	Organization string `json:"organization"`
}

// global model meta information 
// k: 'global' v: GlobalMetaInfo
type GlobalModelMetaInfo struct {
	CurHashId string `json:"curHashId"`
	Round uint32 `json:"round"`
	UploadCount uint32 `json:"uploadCount"`
	TriggerAvgNum uint32 `json:"triggerAvgNum"`

	LocalModelUrls []string `json:"localModelUrls"`
	LocalModelBlocks []string `json:"localModelBlocks"`
}

// meta infomation of local model in each organization 
// k: 'orgX' v: LocalModelMetaInfo
type LocalModelMetaInfo struct{
	CurHashId string `json:"curHashId"`
	//Cert string `json:"cert"`
	//Signature string `json:"signature"`
}

func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	var globalModelMetaInfo GlobalModelMetaInfo = GlobalModelMetaInfo{CurHashId:"0", Round:0, UploadCount:0, TriggerAvgNum:3, LocalModelUrls:[]string{}, LocalModelBlocks:[]string{}}
	globalModelMetaInfoBytes, err := json.Marshal(globalModelMetaInfo)
	err = ctx.GetStub().PutState("global",globalModelMetaInfoBytes)

	timeStr := time.Now().Format("2022-06-29 15:49:05")
	var modelBlock ModelBlock = ModelBlock{ModelType:"global", PrevHashId:"", ModelUrl:"/models/server/model.pth", Timestamp:timeStr, Organization:"Public"}
	modelBlockBytes, err := json.Marshal(modelBlock)
	err = ctx.GetStub().PutState("0", modelBlockBytes)

	for i := 1; i <= 10; i++{
		var localModelMetaInfo LocalModelMetaInfo = LocalModelMetaInfo{CurHashId:""}
		localModelMetaInfoBytes, _ := json.Marshal(localModelMetaInfo)
		err = ctx.GetStub().PutState("org"+strconv.Itoa(i), localModelMetaInfoBytes)
	}
	return err
}

//#########################################Read##################


func (s *SmartContract) GetGlobalModelMetaInfo(ctx contractapi.TransactionContextInterface) (*GlobalModelMetaInfo,error) {
	globalModelMetaInfoBytes, err := ctx.GetStub().GetState("global")

	var globalModelMetaInfo GlobalModelMetaInfo
	_ = json.Unmarshal(globalModelMetaInfoBytes, &globalModelMetaInfo)
	return &globalModelMetaInfo,err
}

func (s *SmartContract) GetLocalModelMetaInfo(ctx contractapi.TransactionContextInterface,org string) (*LocalModelMetaInfo,error) {
	localModelMetaInfoBytes, err := ctx.GetStub().GetState(org)

	var localModelMetaInfo LocalModelMetaInfo
	_ = json.Unmarshal(localModelMetaInfoBytes, &localModelMetaInfo)
	return &localModelMetaInfo,err
}

func (s *SmartContract) GetModelBlock(ctx contractapi.TransactionContextInterface,curHashId string) (*ModelBlock,error) {
	modelBlockBytes, err := ctx.GetStub().GetState(curHashId)

	var modelBlock ModelBlock
	_ = json.Unmarshal(modelBlockBytes, &modelBlock)
	return &modelBlock,err
}

//#########################################Write##################

func (s *SmartContract) UpdateLocalModel(ctx contractapi.TransactionContextInterface, org string, curHashId string, modelUrl string, timestamp string) error{
	localModelMetaInfo, err := s.GetLocalModelMetaInfo(ctx, org)

	var modelBlock ModelBlock = ModelBlock{ModelType:"local", PrevHashId:localModelMetaInfo.CurHashId, ModelUrl:modelUrl, Timestamp:timestamp, Organization:org}
	modelBlockBytes, err := json.Marshal(modelBlock)
	err = ctx.GetStub().PutState(curHashId, modelBlockBytes)

	localModelMetaInfo.CurHashId = curHashId
	localModelMetaInfoBytes, err := json.Marshal(localModelMetaInfo)
	err = ctx.GetStub().PutState(org, localModelMetaInfoBytes)

	globalModelMetaInfo, err := s.GetGlobalModelMetaInfo(ctx)
	globalModelMetaInfo.UploadCount += 1
	globalModelMetaInfo.LocalModelUrls = append(globalModelMetaInfo.LocalModelUrls, modelUrl)
	globalModelMetaInfo.LocalModelBlocks = append(globalModelMetaInfo.LocalModelBlocks, curHashId)
	globalModelMetaInfoBytes, err := json.Marshal(globalModelMetaInfo)
	err = ctx.GetStub().PutState("global",globalModelMetaInfoBytes)
	return err
}

func (s *SmartContract) UpdateGlobalModel(ctx contractapi.TransactionContextInterface, curHashId string, modelUrl string, timestamp string) error{
	globalModelMetaInfo, err := s.GetGlobalModelMetaInfo(ctx)

	var modelBlock ModelBlock = ModelBlock{ModelType:"global", PrevHashId:globalModelMetaInfo.CurHashId, ModelUrl:modelUrl, Timestamp:timestamp, Organization:"Public"}
	modelBlockBytes, err := json.Marshal(modelBlock)
	err = ctx.GetStub().PutState(curHashId, modelBlockBytes)

	globalModelMetaInfo.CurHashId = curHashId
	globalModelMetaInfo.Round += 1
	globalModelMetaInfo.UploadCount = 0
	globalModelMetaInfo.LocalModelUrls = []string{}
	globalModelMetaInfo.LocalModelBlocks = []string{}

	globalModelMetaInfoBytes, err := json.Marshal(globalModelMetaInfo)
	err = ctx.GetStub().PutState("global",globalModelMetaInfoBytes)
	return err
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