package main

import (
	"log"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"time"

	"github.com/faiface/beep"
	"github.com/faiface/beep/effects"
	"github.com/faiface/beep/speaker"
	bwav "github.com/faiface/beep/wav"

	"github.com/mjibson/go-dsp/dsputils"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/wav"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("%+v", err)
	}
}

func run() error {
	const f = "data/cheetah/ENGINE_IDLE.wav"
	rpm, err := computeRPM(f, 8000)
	if err != nil {
		return err
	}

	if err := playAt(f, rpm, 6500); err != nil {
		return err
	}

	return nil
}

type RPMProvider interface {
	Step() int
	Done() bool
}

type JitterRPM struct {
	Provider     RPMProvider
	JitterStdDev int
}

func (jr *JitterRPM) Step() int {
	rpm := jr.Provider.Step()
	rpm += int(rand.NormFloat64() * float64(jr.JitterStdDev))
	return rpm
}

func (jr *JitterRPM) Done() bool {
	return jr.Provider.Done()
}

type EMARPM struct {
	Provider RPMProvider
	Alpha    float64

	lastRPM float64
}

func (r *EMARPM) Step() int {
	rpm := r.Provider.Step()
	newRPM := float64(rpm)*r.Alpha + (1-r.Alpha)*r.lastRPM
	r.lastRPM = newRPM
	return int(newRPM)
}

func (r *EMARPM) Done() bool {
	return r.Provider.Done()
}

type InterpolateRPM struct {
	Start, End, Steps int

	current int
}

func (ir *InterpolateRPM) Step() int {
	stepSize := (ir.End - ir.Start) / ir.Steps
	rpm := ir.Start + stepSize*ir.current
	ir.current += 1
	return rpm
}

func (ir *InterpolateRPM) Done() bool {
	return ir.current >= ir.Steps
}

type SeqRPM struct {
	Providers []RPMProvider

	current int
}

func (sr *SeqRPM) Step() int {
	p := sr.Providers[sr.current]
	rpm := p.Step()
	if p.Done() {
		sr.current += 1
	}
	return rpm
}

func (sr *SeqRPM) Done() bool {
	return sr.current >= len(sr.Providers)
}

func playAt(file string, origRPM, newRPM int) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}
	defer f.Close()

	s, format, err := bwav.Decode(f)
	if err != nil {
		return err
	}
	updateRate := 5 * time.Millisecond
	if err := speaker.Init(format.SampleRate, format.SampleRate.N(updateRate)); err != nil {
		return err
	}
	loop := beep.Loop(-1, s)
	first := true
	resampled := beep.ResampleRatio(10, 1.0, loop)
	gain := effects.Gain{
		Streamer: resampled,
		Gain:     0,
	}
	rpmProvider := EMARPM{
		Provider: &SeqRPM{
			Providers: []RPMProvider{
				&InterpolateRPM{
					Start: 1300, End: 6500, Steps: 1000,
				},
				&InterpolateRPM{
					Start: 6500, End: 6500, Steps: 500,
				},
				&InterpolateRPM{
					Start: 6500, End: 1300, Steps: 1000,
				},
				&InterpolateRPM{
					Start: 1300, End: 1300, Steps: 500,
				},
			},
		},
		Alpha: 0.01,
	}
	for !rpmProvider.Done() {
		newRPM := rpmProvider.Step()
		log.Printf("playing at %d", newRPM)
		ratio := float64(newRPM) / float64(origRPM)
		speaker.Lock()
		resampled.SetRatio(ratio)
		gain.Gain = ratio - 1
		speaker.Unlock()
		if first {
			speaker.Play(&gain)
			first = false
		}
		<-time.NewTimer(updateRate).C
	}
	return nil
}

func f64ToF32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func f32ToF64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func computeRPM(file string, max int) (int, error) {
	f, err := os.Open(file)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	w, err := wav.New(f)
	if err != nil {
		return 0, err
	}
	windowSize := int(w.Samples)
	if int(w.SampleRate) < windowSize {
		windowSize = int(w.SampleRate)
	}
	analysis := make([]complex128, windowSize)
	for i := 0; i < w.Samples/windowSize; i++ {
		data, err := w.ReadFloats(windowSize)
		if err != nil {
			return 0, err
		}
		out := fft.FFTReal(f32ToF64(data))
		for j, v := range out {
			analysis[j] += v
		}
	}
	analysis = analysis[0 : len(analysis)/2]

	var maxRPM, maxValue float64

	var points plotter.XYs
	for i, freq := range analysis {
		// skip 0Hz
		if i == 0 {
			continue
		}

		r, theta := cmplx.Polar(freq)
		theta *= 360.0 / (2 * math.Pi)
		if dsputils.Float64Equal(r, 0) {
			theta = 0 // (When the magnitude is close to 0, the angle is meaningless)
		}
		//fmt.Printf("X(%d) = %.1f ∠ %.1f°\n", i, r, theta)

		frequency := float64(w.SampleRate) / float64(windowSize) * float64(i)
		rpm := frequency * 60

		if rpm > float64(max) {
			break
		}

		points = append(points, plotter.XY{
			X: rpm,
			Y: float64(r),
		})

		if float64(r) > maxValue {
			maxValue = float64(r)
			maxRPM = rpm
		}
	}

	p, err := plot.New()
	if err != nil {
		return 0, err
	}

	p.Title.Text = "FFT Analysis"
	p.X.Label.Text = "RPM"
	p.Y.Label.Text = "Magnitude"

	if err := plotutil.AddLinePoints(p, "First", points); err != nil {
		return 0, err
	}

	if err := p.Save(12*vg.Inch, 4*vg.Inch, "points.png"); err != nil {
		return 0, err
	}

	log.Printf("%s: max RPM %f", file, maxRPM)

	return int(maxRPM), nil
}
