package main

import (
	"strconv"

	rngforest "github.com/malaschitz/randomForest"
)

func main() {
	//allPassengers := GetPassengerData("./in/train.csv", false)
	trainingPassengers := GetPassengerData("./in/train.csv", false)
	testingPassengers := GetPassengerData("./in/test.csv", true)
	// split into training training and testing training groups
	// so I can know how good it's doing
	// 10% or so for testing
	// uncomment these lines for that
	//trainingPassengers := make([]Passenger, 0)
	//testingPassengers := make([]Passenger, 0)
	//for _, passenger := range allPassengers {
	//	if rand.Float64()*100 > 10 {
	//		trainingPassengers = append(trainingPassengers, passenger)
	//	} else {
	//		testingPassengers = append(testingPassengers, passenger)
	//	}
	//}
	forest := rngforest.Forest{}
	forest.Data = GetForestDataFromPassengers(trainingPassengers)
	forest.Train(10000)
	out := [][]string{{"PassengerId", "Survived"}}
	for _, passenger := range testingPassengers {
		vote := forest.Vote(GetXDataFromPassenger(passenger))
		survived := "0"
		if vote[0] < vote[1] {
			survived = "1"
		}
		id := strconv.FormatInt(int64(passenger.Id), 10)
		out = append(out, []string{
			id,
			survived,
		})
	}
	SaveCSV(out)
}
