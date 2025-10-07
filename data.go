package main

import (
	"encoding/csv"
	"errors"
	"log"
	"os"
	"strconv"

	"github.com/davecgh/go-spew/spew"
	rngforest "github.com/malaschitz/randomForest"
)

const (
	avgAge  float64 = 29.7
	avgFare float64 = 32.2
)

func OpenCSV(path string) [][]string {
	file, err := os.Open(path)
	if err != nil {
		log.Fatalf("Unable to open csv file: %v. Error was: %v", path, err.Error())
	}
	defer file.Close()

	data, err := csv.NewReader(file).ReadAll()
	if err != nil {
		log.Fatalf("Unable to read csv file: %v. Error was: %v", path, err.Error())
	}

	return data
}

func SaveCSV(out [][]string) string {
	file, _ := os.CreateTemp("./out", "titanic_trees*.csv")
	defer file.Close()

	w := csv.NewWriter(file)
	w.WriteAll(out)
	w.Flush()

	return file.Name()
}

// Get Passengers from the csv data provided from path. Test data does not
// contain survivor information, so it will parse the csv data differently
// than with training data.
func GetPassengerData(path string, testdata bool) []Passenger {
	ret := make([]Passenger, 0)
	csv := OpenCSV(path)
	for i, row := range csv {
		if i == 0 {
			spew.Dump(row)
			continue // first row is label
		}
		p, err := NewPassenger(row, testdata)
		if err != nil {
			log.Printf("Uh oh. Stinky. %v", err.Error())
		}
		ret = append(ret, p)
	}
	return ret
}

// Passenger is a titanic passenger
type Passenger struct {
	Id       int
	Survived bool
	Class    Class
	Name     string
	Age      float64
	Sex      Sex
	SibSp    SibSip
	Parch    Parch
	Ticket   string
	Fare     float64
	Cabin    string
	Embarked string
}

func NewPassenger(data []string, testdata bool) (p Passenger, err error) {
	if len(data) < 10 {
		err = errors.New("Unable to parse passenger data; not enough data!")
	}
	i := 0
	if p.Id, err = strconv.Atoi(data[i]); err != nil {
		return
	}
	i++
	if !testdata {
		var c int
		if c, err = strconv.Atoi(data[i]); err != nil {
			return
		}
		p.Survived = c == 1
		i++
	}
	switch data[i] {
	case "1":
		p.Class = Class1
	case "2":
		p.Class = Class2
	case "3":
		p.Class = Class3
	default:
		err = errors.New("Unable to parse Class for this passenger")
		return
	}
	i++
	p.Name = data[i]
	i++
	switch data[i] {
	case "male":
		p.Sex = Male
	case "female":
		p.Sex = Female
	default:
		err = errors.New("Unable to parse Sex for this passenger.")
		return
	}
	i++
	if data[i] == "" {
		p.Age = avgAge
	} else {
		if p.Age, err = strconv.ParseFloat(data[i], 64); err != nil {
			return
		}
	}
	i++
	switch data[i] {
	case "0":
		p.SibSp = SibSip0
	case "1":
		p.SibSp = SibSip1
	case "2":
		p.SibSp = SibSip2
	case "3":
		p.SibSp = SibSip3
	case "4":
		p.SibSp = SibSip4
	case "5":
		p.SibSp = SibSip5
	case "6":
		p.SibSp = SibSip6
	case "7":
		p.SibSp = SibSip7
	case "8":
		p.SibSp = SibSip8
	default:
		err = errors.New("Unable to parse SibSip")
		return
	}
	i++
	switch data[i] {
	case "0":
		p.Parch = Parch0
	case "1":
		p.Parch = Parch1
	case "2":
		p.Parch = Parch2
	case "3":
		p.Parch = Parch3
	case "4":
		p.Parch = Parch4
	case "5":
		p.Parch = Parch5
	case "6":
		p.Parch = Parch6
	case "7":
		p.Parch = Parch7
	case "8":
		p.Parch = Parch8
	case "9":
		p.Parch = Parch9
	default:
		err = errors.New("Unable to parse parch")
		return
	}
	i++
	p.Ticket = data[i]
	i++
	if data[i] == "" {
		p.Fare = avgFare
	} else {
		if p.Fare, err = strconv.ParseFloat(data[i], 64); err != nil {
			return
		}
	}
	i++
	p.Cabin = data[i]
	i++
	p.Embarked = data[i]
	i++
	return
}

type Class uint

const (
	Class1 Class = iota
	Class2
	Class3
)

type Sex uint

const (
	Male Sex = iota
	Female
)

type SibSip uint

const (
	SibSip0 SibSip = iota
	SibSip1
	SibSip2
	SibSip3
	SibSip4
	SibSip5
	SibSip6
	SibSip7
	SibSip8
)

type Parch uint

const (
	Parch0 Parch = iota
	Parch1
	Parch2
	Parch3
	Parch4
	Parch5
	Parch6
	Parch7
	Parch8
	Parch9
)

// GetForestDataFromPassengers generates the forest data
// from a slice of passengers.
// The Class data is if they survived, since that's what we
// want to predict.
// The tree data is:
// // Age
// // PClass
// // Fare
// // Parch
// // Sex
// // SibSp
func GetForestDataFromPassengers(passengers []Passenger) rngforest.ForestData {
	x := make([][]float64, 0)
	c := make([]int, 0)
	for _, passenger := range passengers {
		if passenger.Survived {
			c = append(c, 1)
		} else {
			c = append(c, 0)
		}
		x = append(x, GetXDataFromPassenger(passenger))
	}
	return rngforest.ForestData{
		X:     x,
		Class: c,
	}
}

func GetXDataFromPassenger(p Passenger) []float64 {
	return []float64{
		p.Age,
		float64(p.Class),
		p.Fare,
		float64(p.Parch),
		float64(p.Sex),
		float64(p.SibSp),
	}
}
