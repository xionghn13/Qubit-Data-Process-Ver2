I0 = 5.27
hI = 2.57  # mA
I = 5.6943
flux = (I - I0) / (hI * 2)
print('flux = %.3G' % flux)

flux = -0.0286
I = flux * hI * 2 + I0
print('I = %.3G' % I)
