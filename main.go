package main

import (
	"fmt"
	"github.com/Kagami/go-face"
	"log"
	"path/filepath"
)

const dataDir = "images"

func main() {
	fmt.Println("Initializing facial recognition")

	recognizer, err := face.NewRecognizer(dataDir)
	if err != nil {
		fmt.Println("Failed to initialized recognizer")
		fmt.Println(err)
	}
	defer recognizer.Close()

	avengerFile := filepath.Join(dataDir, "avengers-02.jpeg")
	faces, err := recognizer.RecognizeFile(avengerFile)
	if err != nil {
		log.Fatalf("Cannot recognize face")
	}

	fmt.Println("Number of faces: ", len(faces))

	var samples []face.Descriptor
	var avengers []int32

	for i, f := range faces {
		samples = append(samples, f.Descriptor)
		avengers = append(avengers, int32(i))
	}

	labels := []string{
		"Dr Strange",
		"Tony Stark",
		"Bruce Banner",
		"Wong",
	}

	recognizer.SetSamples(samples, avengers)

	testTonyStark := filepath.Join(dataDir, "dr-strange.jpg")
	tonyStark, err := recognizer.RecognizeSingleFile(testTonyStark)
	if err != nil {
		log.Fatal("Faced error with file: %v", err)
	}
	avengersId := recognizer.Classify(tonyStark.Descriptor)
	if avengersId < 0 {
		log.Fatal("Can't classify from existing data")
	}

	fmt.Println(avengersId)
	fmt.Println(labels[avengersId])
}
