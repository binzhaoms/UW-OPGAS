a)
Parabolic fit coefficients: A=0.184935, B=-5.264235, C=88.295455
E2 error: 172.359276

Fitted values vector length: 2301
First 10 fitted values: [83.21615385 83.16722868 83.1183405  83.06948931 83.0206751  82.97189788
 82.92315765 82.8744544  82.82578814 82.77715887]
Last 10 fitted values: [68.15251539 68.18832727 68.22417614 68.26006199 68.29598484 68.33194466
 68.36794148 68.40397528 68.44004607 68.47615385]

Linear interpolation vector length: 2301
First 10 linear interpolated values: [75.   75.02 75.04 75.06 75.08 75.1  75.12 75.14 75.16 75.18]
Last 10 linear interpolated values: [71.55 71.6  71.65 71.7  71.75 71.8  71.85 71.9  71.95 72.  ]

b)
Spline interpolation vector length: 2301
First 10 spline interpolated values: [75.         75.03975189 75.07905598 75.11791378 75.15632676 75.19429642
 75.23182426 75.26891177 75.30556044 75.34177177]
Last 10 spline interpolated values: [71.4699681  71.5278949  71.58605961 71.64446373 71.70310876 71.7619962
 71.82112754 71.88050429 71.94012794 72.        ]

c)
Started with
    p0 = [10, 0.5, 60]
    E2 = 2015.96
Result is way off.
Brut force a new series of A,B,C and check with one is playing a bigger role in minimize the E2.
    # A variants
    # p1 = [5, 0.5, 60]
    # p2 = [20, 0.5, 60]
    No big difference. P0 wins with E2=2016.0

    # B variants
    # p1 = [10, 0.1, 60]
    # p2 = [10, 1.0, 60]
    Now we realize that B is the key factor and with a smaller B=0.1 we get E2=37.6 which is very close to the input data set.

    Use B=0.1 and try different C to see if it can converge better
    # C variants
    p1 = [10, 0.1, 0]
    p2 = [10, 0.1, 100]
    Not making any difference. We seem get enough evidence that E=37.6 is the right converge.

Take a step back and take a look at the curve and the curve
y = A*cos(B*x) + C
We can notice that from the initial bad guess we expect to see a much "fatter" curve to make sure the waves are matching. And A is ctaullt controlling the height and C is controlling the base line. We can see that they all converge good. So B seems to be the key factor here we want to tweat with.

