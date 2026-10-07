:- discontiguous model/2, item/3, answer/6, axis_p/4, item_div/5, level/2, token_stat/5, token_rate/4, boundary_token/2, lexicon/2, dbn_unit/4, threshold/2.
threshold(item_jsd, 0.150000).
threshold(log_bf, 1.000000).
threshold(token_mi, 0.100000).
model('gemma4-e4b', 2).
model('gemma4-e2b', 1).
item('i01', 'rule_banned', 'b').
item('i02', 'rule_banned', 'b').
item('i03', 'rule_banned', 'b').
item('i04', 'rule_banned', 'b').
item('i05', 'rule_banned', 'b').
item('i06', 'rule_banned', 'b').
item('i07', 'rule_banned', 'b').
item('i08', 'rule_banned', 'b').
item('i09', 'rule_banned', 'b').
item('i10', 'rule_banned', 'b').
item('i11', 'rule_banned', 'b').
item('i12', 'rule_banned', 'b').
item('i13', 'rule_allowed', 'a').
item('i14', 'rule_allowed', 'a').
item('i15', 'rule_allowed', 'a').
item('i16', 'rule_allowed', 'a').
item('i17', 'rule_allowed', 'a').
item('i18', 'rule_allowed', 'a').
item('i19', 'syntax', 'none').
item('i20', 'syntax', 'none').
item('i21', 'syntax', 'a').
item('i22', 'syntax', 'none').
item('i23', 'syntax', 'none').
item('i24', 'error', 'none').
item('i25', 'error', 'none').
item('i26', 'error', 'none').
item('i27', 'disputed', 'none').
item('i28', 'disputed', 'none').
item('i29', 'disputed', 'none').
item('i30', 'disputed', 'none').
item('i31', 'disputed', 'none').
item('i32', 'disputed', 'none').
item('i33', 'disputed', 'none').
item('i34', 'disputed', 'none').
item('i35', 'disputed', 'none').
item('i36', 'disputed', 'none').
item('i37', 'disputed', 'none').
item('i38', 'disputed', 'none').
item_div('i01', 0.279707, 2.807956, 0.096979, 0.134029).
answer('gemma4-e4b', 'i01', 0.386279, 0.200544, 0.304688, 0.108488).
axis_p('gemma4-e4b', 'i01', 0.596471, 0.505814).
answer('gemma4-e2b', 'i01', 0.077267, 0.787048, 0.064055, 0.071630).
axis_p('gemma4-e2b', 'i01', 0.904794, 0.890115).
item_div('i02', 0.385887, 4.358078, 0.362253, 0.011125).
answer('gemma4-e4b', 'i02', 0.121132, 0.137474, 0.637077, 0.104317).
axis_p('gemma4-e4b', 'i02', 0.231785, 0.805056).
answer('gemma4-e2b', 'i02', 0.069753, 0.790163, 0.063876, 0.076208).
axis_p('gemma4-e2b', 'i02', 0.899907, 0.893377).
item_div('i03', 0.242130, 2.115865, 0.063844, 0.158519).
answer('gemma4-e4b', 'i03', 0.437180, 0.225600, 0.226653, 0.110568).
axis_p('gemma4-e4b', 'i03', 0.680866, 0.446948).
answer('gemma4-e2b', 'i03', 0.094952, 0.777670, 0.062107, 0.065271).
axis_p('gemma4-e2b', 'i03', 0.914025, 0.877530).
item_div('i04', 0.112774, -0.016597, 0.024176, 0.018393).
answer('gemma4-e4b', 'i04', 0.127090, 0.175482, 0.628077, 0.069351).
axis_p('gemma4-e4b', 'i04', 0.280635, 0.837287).
answer('gemma4-e2b', 'i04', 0.076203, 0.384765, 0.298623, 0.240409).
axis_p('gemma4-e2b', 'i04', 0.456632, 0.703764).
item_div('i05', 0.202447, 1.495325, 0.088539, 0.093043).
answer('gemma4-e4b', 'i05', 0.419420, 0.168717, 0.255881, 0.155982).
axis_p('gemma4-e4b', 'i05', 0.597930, 0.416219).
answer('gemma4-e2b', 'i05', 0.184052, 0.671571, 0.066932, 0.077445).
axis_p('gemma4-e2b', 'i05', 0.895136, 0.765003).
item_div('i06', 0.228836, 1.668456, 0.018010, 0.191139).
answer('gemma4-e4b', 'i06', 0.547976, 0.223386, 0.149516, 0.079122).
axis_p('gemma4-e4b', 'i06', 0.801513, 0.358781).
answer('gemma4-e2b', 'i06', 0.120009, 0.750025, 0.063607, 0.066358).
axis_p('gemma4-e2b', 'i06', 0.911150, 0.848481).
item_div('i07', 0.387668, 4.076766, 0.001288, 0.386310).
answer('gemma4-e4b', 'i07', 0.745170, 0.104782, 0.086254, 0.063794).
axis_p('gemma4-e4b', 'i07', 0.888835, 0.156707).
answer('gemma4-e2b', 'i07', 0.115714, 0.756881, 0.061324, 0.066081).
axis_p('gemma4-e2b', 'i07', 0.913994, 0.853561).
item_div('i08', 0.268708, 2.370144, 0.001921, 0.198153).
answer('gemma4-e4b', 'i08', 0.681783, 0.134727, 0.116190, 0.067300).
axis_p('gemma4-e4b', 'i08', 0.851678, 0.223241).
answer('gemma4-e2b', 'i08', 0.151371, 0.630475, 0.080495, 0.137660).
axis_p('gemma4-e2b', 'i08', 0.813162, 0.734411).
item_div('i09', 0.245731, 2.424557, 0.000951, 0.003332).
answer('gemma4-e4b', 'i09', 0.318371, 0.176839, 0.410018, 0.094772).
axis_p('gemma4-e4b', 'i09', 0.494678, 0.596508).
answer('gemma4-e2b', 'i09', 0.086471, 0.441396, 0.084802, 0.387331).
axis_p('gemma4-e2b', 'i09', 0.530964, 0.529109).
item_div('i10', 0.286099, 2.191348, 0.021019, 0.243493).
answer('gemma4-e4b', 'i10', 0.558240, 0.189707, 0.185805, 0.066248).
axis_p('gemma4-e4b', 'i10', 0.775496, 0.361680).
answer('gemma4-e2b', 'i10', 0.083333, 0.750000, 0.083333, 0.083333).
axis_p('gemma4-e2b', 'i10', 0.900000, 0.900000).
item_div('i11', 0.183457, 1.042272, 0.020264, 0.124821).
answer('gemma4-e4b', 'i11', 0.434653, 0.282275, 0.203913, 0.079159).
axis_p('gemma4-e4b', 'i11', 0.741031, 0.484653).
answer('gemma4-e2b', 'i11', 0.091526, 0.743569, 0.084996, 0.079909).
axis_p('gemma4-e2b', 'i11', 0.872328, 0.865073).
item_div('i12', 0.220743, 1.575991, 0.023913, 0.161858).
answer('gemma4-e4b', 'i12', 0.498418, 0.253703, 0.182629, 0.065251).
axis_p('gemma4-e4b', 'i12', 0.780134, 0.429258).
answer('gemma4-e2b', 'i12', 0.102446, 0.766676, 0.065284, 0.065594).
axis_p('gemma4-e2b', 'i12', 0.910135, 0.868844).
item_div('i13', 0.183918, 0.770699, 0.001990, 0.170980).
answer('gemma4-e4b', 'i13', 0.755513, 0.097875, 0.090010, 0.056602).
axis_p('gemma4-e4b', 'i13', 0.892654, 0.153206).
answer('gemma4-e2b', 'i13', 0.337905, 0.542777, 0.060381, 0.058936).
axis_p('gemma4-e2b', 'i13', 0.922981, 0.614621).
item_div('i14', 0.246918, 1.807596, 0.000922, 0.231041).
answer('gemma4-e4b', 'i14', 0.721521, 0.119791, 0.096570, 0.062117).
axis_p('gemma4-e4b', 'i14', 0.879236, 0.184846).
answer('gemma4-e2b', 'i14', 0.218368, 0.643032, 0.066413, 0.072187).
axis_p('gemma4-e2b', 'i14', 0.901555, 0.732717).
item_div('i15', 0.034750, -1.651681, 0.000092, 0.030219).
answer('gemma4-e4b', 'i15', 0.760543, 0.105348, 0.075371, 0.058739).
axis_p('gemma4-e4b', 'i15', 0.906545, 0.145243).
answer('gemma4-e2b', 'i15', 0.600127, 0.271587, 0.062866, 0.065420).
axis_p('gemma4-e2b', 'i15', 0.913016, 0.316059).
item_div('i16', 0.401286, 3.302973, 0.000079, 0.430968).
answer('gemma4-e4b', 'i16', 0.735764, 0.118496, 0.086058, 0.059682).
axis_p('gemma4-e4b', 'i16', 0.893622, 0.171727).
answer('gemma4-e2b', 'i16', 0.083333, 0.750000, 0.083333, 0.083333).
axis_p('gemma4-e2b', 'i16', 0.900000, 0.900000).
item_div('i17', 0.296033, 2.625298, 0.002851, 0.268673).
answer('gemma4-e4b', 'i17', 0.708351, 0.119799, 0.110910, 0.060940).
axis_p('gemma4-e4b', 'i17', 0.864612, 0.200788).
answer('gemma4-e2b', 'i17', 0.168061, 0.696155, 0.065982, 0.069802).
axis_p('gemma4-e2b', 'i17', 0.904685, 0.791263).
item_div('i18', 0.105976, -0.446467, 0.000350, 0.097219).
answer('gemma4-e4b', 'i18', 0.724062, 0.125453, 0.085585, 0.064900).
axis_p('gemma4-e4b', 'i18', 0.888350, 0.178932).
answer('gemma4-e2b', 'i18', 0.405628, 0.456030, 0.065367, 0.072974).
axis_p('gemma4-e2b', 'i18', 0.901843, 0.523775).
item_div('i19', 0.170204, 0.565505, 0.000521, 0.159688).
answer('gemma4-e4b', 'i19', 0.688234, 0.153161, 0.090947, 0.067658).
axis_p('gemma4-e4b', 'i19', 0.879327, 0.215676).
answer('gemma4-e2b', 'i19', 0.266365, 0.590296, 0.066272, 0.077067).
axis_p('gemma4-e2b', 'i19', 0.896290, 0.673964).
item_div('i20', 0.143678, 0.300423, 0.030527, 0.082915).
answer('gemma4-e4b', 'i20', 0.545374, 0.198699, 0.183336, 0.072591).
axis_p('gemma4-e4b', 'i20', 0.771192, 0.368928).
answer('gemma4-e2b', 'i20', 0.255334, 0.620665, 0.062677, 0.061324).
axis_p('gemma4-e2b', 'i20', 0.917777, 0.703713).
item_div('i21', 0.103224, -0.281967, 0.020440, 0.055444).
answer('gemma4-e4b', 'i21', 0.546867, 0.181060, 0.169318, 0.102755).
axis_p('gemma4-e4b', 'i21', 0.753252, 0.333753).
answer('gemma4-e2b', 'i21', 0.314921, 0.528954, 0.068895, 0.087230).
axis_p('gemma4-e2b', 'i21', 0.882083, 0.608721).
item_div('i22', 0.061940, -0.842207, 0.021490, 0.004539).
answer('gemma4-e4b', 'i22', 0.250000, 0.416667, 0.250000, 0.083333).
axis_p('gemma4-e4b', 'i22', 0.700000, 0.700000).
answer('gemma4-e2b', 'i22', 0.149531, 0.659947, 0.083000, 0.107522).
axis_p('gemma4-e2b', 'i22', 0.843864, 0.769941).
item_div('i23', 0.168587, 0.943467, 0.041547, 0.038104).
answer('gemma4-e4b', 'i23', 0.351246, 0.251973, 0.294202, 0.102579).
axis_p('gemma4-e4b', 'i23', 0.614688, 0.551305).
answer('gemma4-e2b', 'i23', 0.123115, 0.671918, 0.069026, 0.135941).
axis_p('gemma4-e2b', 'i23', 0.827815, 0.767715).
item_div('i24', 0.274699, 2.196854, 0.172590, 0.000008).
answer('gemma4-e4b', 'i24', 0.133021, 0.117314, 0.565461, 0.184204).
axis_p('gemma4-e4b', 'i24', 0.222594, 0.703084).
answer('gemma4-e2b', 'i24', 0.083333, 0.583333, 0.083333, 0.250000).
axis_p('gemma4-e2b', 'i24', 0.700000, 0.700000).
item_div('i25', 0.105873, -0.224667, 0.083921, 0.000687).
answer('gemma4-e4b', 'i25', 0.191724, 0.294352, 0.408322, 0.105602).
axis_p('gemma4-e4b', 'i25', 0.484529, 0.725193).
answer('gemma4-e2b', 'i25', 0.162387, 0.613363, 0.113713, 0.110537).
axis_p('gemma4-e2b', 'i25', 0.806389, 0.752307).
item_div('i26', 0.116450, 0.070724, 0.101256, 0.023542).
answer('gemma4-e4b', 'i26', 0.083333, 0.416667, 0.416667, 0.083333).
axis_p('gemma4-e4b', 'i26', 0.500000, 0.900000).
answer('gemma4-e2b', 'i26', 0.152640, 0.657873, 0.082407, 0.107080).
axis_p('gemma4-e2b', 'i26', 0.845015, 0.766978).
item_div('i27', 0.181559, 0.906372, 0.014173, 0.133451).
answer('gemma4-e4b', 'i27', 0.525135, 0.226163, 0.170995, 0.077707).
axis_p('gemma4-e4b', 'i27', 0.779220, 0.385731).
answer('gemma4-e2b', 'i27', 0.151679, 0.693446, 0.077094, 0.077781).
axis_p('gemma4-e2b', 'i27', 0.883472, 0.800601).
item_div('i28', 0.208910, 1.350980, 0.019920, 0.151739).
answer('gemma4-e4b', 'i28', 0.532679, 0.221035, 0.175683, 0.070602).
axis_p('gemma4-e4b', 'i28', 0.781905, 0.385243).
answer('gemma4-e2b', 'i28', 0.137961, 0.723725, 0.067886, 0.070427).
axis_p('gemma4-e2b', 'i28', 0.901874, 0.824013).
item_div('i29', 0.196446, 1.242136, 0.022653, 0.110262).
answer('gemma4-e4b', 'i29', 0.542256, 0.170138, 0.201747, 0.085860).
axis_p('gemma4-e4b', 'i29', 0.735993, 0.357650).
answer('gemma4-e2b', 'i29', 0.189021, 0.648489, 0.068802, 0.093688).
axis_p('gemma4-e2b', 'i29', 0.875011, 0.741434).
item_div('i30', 0.252059, 1.922814, 0.000738, 0.229671).
answer('gemma4-e4b', 'i30', 0.719869, 0.117576, 0.099333, 0.063222).
axis_p('gemma4-e4b', 'i30', 0.874939, 0.185454).
answer('gemma4-e2b', 'i30', 0.212903, 0.642896, 0.065705, 0.078496).
axis_p('gemma4-e2b', 'i30', 0.895332, 0.731779).
item_div('i31', 0.265353, 2.058087, 0.001653, 0.259411).
answer('gemma4-e4b', 'i31', 0.728111, 0.118728, 0.090358, 0.062803).
axis_p('gemma4-e4b', 'i31', 0.885377, 0.176762).
answer('gemma4-e2b', 'i31', 0.205965, 0.666717, 0.063644, 0.063674).
axis_p('gemma4-e2b', 'i31', 0.914091, 0.755957).
item_div('i32', 0.152173, 0.540940, 0.025668, 0.099342).
answer('gemma4-e4b', 'i32', 0.400783, 0.327561, 0.190149, 0.081506).
axis_p('gemma4-e4b', 'i32', 0.753716, 0.519678).
answer('gemma4-e2b', 'i32', 0.102607, 0.753329, 0.067624, 0.076440).
axis_p('gemma4-e2b', 'i32', 0.895485, 0.856614).
item_div('i33', 0.212256, 1.334250, 0.003915, 0.180037).
answer('gemma4-e4b', 'i33', 0.661580, 0.142408, 0.121550, 0.074462).
axis_p('gemma4-e4b', 'i33', 0.837765, 0.237732).
answer('gemma4-e2b', 'i33', 0.215561, 0.633907, 0.069711, 0.080821).
axis_p('gemma4-e2b', 'i33', 0.888298, 0.726242).
item_div('i34', 0.354432, 3.568763, 0.000733, 0.332061).
answer('gemma4-e4b', 'i34', 0.720889, 0.122254, 0.100240, 0.056617).
axis_p('gemma4-e4b', 'i34', 0.881270, 0.191660).
answer('gemma4-e2b', 'i34', 0.118794, 0.742205, 0.065669, 0.073332).
axis_p('gemma4-e2b', 'i34', 0.901110, 0.842082).
item_div('i35', 0.163527, 0.655549, 0.027100, 0.090129).
answer('gemma4-e4b', 'i35', 0.500903, 0.218207, 0.207738, 0.073151).
axis_p('gemma4-e4b', 'i35', 0.743456, 0.417718).
answer('gemma4-e2b', 'i35', 0.188179, 0.664103, 0.071059, 0.076659).
axis_p('gemma4-e2b', 'i35', 0.891424, 0.761291).
item_div('i36', 0.389933, 2.617380, 0.480084, 0.145653).
answer('gemma4-e4b', 'i36', 0.802753, 0.081517, 0.059540, 0.056190).
axis_p('gemma4-e4b', 'i36', 0.926967, 0.101175).
answer('gemma4-e2b', 'i36', 0.125000, 0.125000, 0.375000, 0.375000).
axis_p('gemma4-e2b', 'i36', 0.166667, 0.500000).
item_div('i37', 0.195739, 1.379086, 0.044278, 0.060251).
answer('gemma4-e4b', 'i37', 0.359228, 0.254506, 0.300224, 0.086042).
axis_p('gemma4-e4b', 'i37', 0.626371, 0.560811).
answer('gemma4-e2b', 'i37', 0.092831, 0.715620, 0.076235, 0.115313).
axis_p('gemma4-e2b', 'i37', 0.842724, 0.824284).
item_div('i38', 0.204582, 1.247102, 0.017571, 0.156508).
answer('gemma4-e4b', 'i38', 0.578014, 0.192723, 0.155511, 0.073753).
axis_p('gemma4-e4b', 'i38', 0.800819, 0.331371).
answer('gemma4-e2b', 'i38', 0.175401, 0.693164, 0.063248, 0.068187).
axis_p('gemma4-e2b', 'i38', 0.909517, 0.784902).
level(0, 0.095058).
boundary_token(0, 'implementation').
token_stat(0, 'implementation', 0.295807, 2.051271, 3).
token_rate(0, 'gemma4-e4b', 'implementation', 0.100000).
token_rate(0, 'gemma4-e2b', 'implementation', 0.700000).
boundary_token(0, 'international').
token_stat(0, 'international', 0.295807, 2.051271, 3).
token_rate(0, 'gemma4-e4b', 'international', 0.100000).
token_rate(0, 'gemma4-e2b', 'international', 0.700000).
boundary_token(0, 'international sanctions').
token_stat(0, 'international sanctions', 0.295807, 2.051271, 3).
token_rate(0, 'gemma4-e4b', 'international sanctions', 0.100000).
token_rate(0, 'gemma4-e2b', 'international sanctions', 0.700000).
lexicon('international sanctions', 'legal').
boundary_token(0, 'legality').
token_stat(0, 'legality', 0.295807, 2.051271, 3).
token_rate(0, 'gemma4-e4b', 'legality', 0.100000).
token_rate(0, 'gemma4-e2b', 'legality', 0.700000).
lexicon('legality', 'legal').
boundary_token(0, 'model').
token_stat(0, 'model', 0.295807, 2.051271, 5).
token_rate(0, 'gemma4-e4b', 'model', 0.900000).
token_rate(0, 'gemma4-e2b', 'model', 0.300000).
boundary_token(0, 'technical').
token_stat(0, 'technical', 0.295807, 2.051271, 3).
token_rate(0, 'gemma4-e4b', 'technical', 0.100000).
token_rate(0, 'gemma4-e2b', 'technical', 0.700000).
dbn_unit(0, 0, 0.188722, ['international sanctions', 'international', 'implementation', 'legality', 'technical', 'significant', 'sanctions list', 'technical implementation']).
dbn_unit(0, 1, 0.188722, ['sanctions list', 'list', 'kaggle pays', 'sanctions', 'kaggle', 'resident', 'implementation', 'significant']).
dbn_unit(0, 2, 0.188722, ['yaml', 'declarative agent', 'agent yaml', 'declarative', 'instead', 'team submits', 'python agent', 'agent py']).
level(1, 0.142659).
boundary_token(1, 'chosen').
token_stat(1, 'chosen', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'chosen', 0.100000).
token_rate(1, 'gemma4-e2b', 'chosen', 0.900000).
boundary_token(1, 'chosen option').
token_stat(1, 'chosen option', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'chosen option', 0.100000).
token_rate(1, 'gemma4-e2b', 'chosen option', 0.900000).
boundary_token(1, 'instances').
token_stat(1, 'instances', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'instances', 0.100000).
token_rate(1, 'gemma4-e2b', 'instances', 0.900000).
boundary_token(1, 'leading').
token_stat(1, 'leading', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'leading', 0.100000).
token_rate(1, 'gemma4-e2b', 'leading', 0.900000).
boundary_token(1, 'option').
token_stat(1, 'option', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'option', 0.100000).
token_rate(1, 'gemma4-e2b', 'option', 0.900000).
boundary_token(1, 'option seems').
token_stat(1, 'option seems', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'option seems', 0.100000).
token_rate(1, 'gemma4-e2b', 'option seems', 0.900000).
boundary_token(1, 'reasoning leading').
token_stat(1, 'reasoning leading', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'reasoning leading', 0.100000).
token_rate(1, 'gemma4-e2b', 'reasoning leading', 0.900000).
boundary_token(1, 'two').
token_stat(1, 'two', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'two', 0.100000).
token_rate(1, 'gemma4-e2b', 'two', 0.900000).
boundary_token(1, 'two models').
token_stat(1, 'two models', 0.531004, 4.248495, 4).
token_rate(1, 'gemma4-e4b', 'two models', 0.100000).
token_rate(1, 'gemma4-e2b', 'two models', 0.900000).
boundary_token(1, 'counter-intuitive').
token_stat(1, 'counter-intuitive', 0.295807, 2.051271, 5).
token_rate(1, 'gemma4-e4b', 'counter-intuitive', 0.300000).
token_rate(1, 'gemma4-e2b', 'counter-intuitive', 0.900000).
boundary_token(1, 'data provided').
token_stat(1, 'data provided', 0.295807, 2.051271, 3).
token_rate(1, 'gemma4-e4b', 'data provided', 0.100000).
token_rate(1, 'gemma4-e2b', 'data provided', 0.700000).
boundary_token(1, 'diverge').
token_stat(1, 'diverge', 0.295807, 2.051271, 5).
token_rate(1, 'gemma4-e4b', 'diverge', 0.300000).
token_rate(1, 'gemma4-e2b', 'diverge', 0.900000).
boundary_token(1, 'diverge significantly').
token_stat(1, 'diverge significantly', 0.295807, 2.051271, 5).
token_rate(1, 'gemma4-e4b', 'diverge significantly', 0.300000).
token_rate(1, 'gemma4-e2b', 'diverge significantly', 0.900000).
boundary_token(1, 'especially').
token_stat(1, 'especially', 0.295807, 2.051271, 3).
token_rate(1, 'gemma4-e4b', 'especially', 0.100000).
token_rate(1, 'gemma4-e2b', 'especially', 0.700000).
boundary_token(1, 'model\'s reasoning').
token_stat(1, 'model\'s reasoning', 0.295807, 2.051271, 5).
token_rate(1, 'gemma4-e4b', 'model\'s reasoning', 0.300000).
token_rate(1, 'gemma4-e2b', 'model\'s reasoning', 0.900000).
boundary_token(1, 'models diverge').
token_stat(1, 'models diverge', 0.295807, 2.051271, 5).
token_rate(1, 'gemma4-e4b', 'models diverge', 0.300000).
token_rate(1, 'gemma4-e2b', 'models diverge', 0.900000).
boundary_token(1, 'seems').
token_stat(1, 'seems', 0.295807, 2.051271, 5).
token_rate(1, 'gemma4-e4b', 'seems', 0.300000).
token_rate(1, 'gemma4-e2b', 'seems', 0.900000).
boundary_token(1, 'specific statement').
token_stat(1, 'specific statement', 0.295807, 2.051271, 5).
token_rate(1, 'gemma4-e4b', 'specific statement', 0.300000).
token_rate(1, 'gemma4-e2b', 'specific statement', 0.900000).
boundary_token(1, 'statement').
token_stat(1, 'statement', 0.295807, 2.051271, 5).
token_rate(1, 'gemma4-e4b', 'statement', 0.300000).
token_rate(1, 'gemma4-e2b', 'statement', 0.900000).
boundary_token(1, 'statement especially').
token_stat(1, 'statement especially', 0.295807, 2.051271, 3).
token_rate(1, 'gemma4-e4b', 'statement especially', 0.100000).
token_rate(1, 'gemma4-e2b', 'statement especially', 0.700000).
dbn_unit(1, 0, 1.000000, ['two models', 'option seems', 'option', 'two', 'leading', 'instances', 'chosen option', 'chosen']).
dbn_unit(1, 1, 1.000000, ['model', 'implementation', 'based', 'provided', 'token', 'technical', 'discussing', 'model m2']).
dbn_unit(1, 2, 1.000000, ['diverge significantly', 'counter-intuitive', 'seems', 'specific statement', 'statement', 'model\'s reasoning', 'models diverge', 'diverge']).
level(2, 0.088512).
boundary_token(2, 'appear').
token_stat(2, 'appear', 0.531004, 4.248495, 4).
token_rate(2, 'gemma4-e4b', 'appear', 0.100000).
token_rate(2, 'gemma4-e2b', 'appear', 0.900000).
boundary_token(2, 'different frequency').
token_stat(2, 'different frequency', 0.531004, 4.248495, 4).
token_rate(2, 'gemma4-e4b', 'different frequency', 0.100000).
token_rate(2, 'gemma4-e2b', 'different frequency', 0.900000).
boundary_token(2, 'chosen').
token_stat(2, 'chosen', 0.295807, 2.051271, 5).
token_rate(2, 'gemma4-e4b', 'chosen', 0.900000).
token_rate(2, 'gemma4-e2b', 'chosen', 0.300000).
boundary_token(2, 'chosen option').
token_stat(2, 'chosen option', 0.295807, 2.051271, 5).
token_rate(2, 'gemma4-e4b', 'chosen option', 0.900000).
token_rate(2, 'gemma4-e2b', 'chosen option', 0.300000).
boundary_token(2, 'different').
token_stat(2, 'different', 0.295807, 2.051271, 5).
token_rate(2, 'gemma4-e4b', 'different', 0.300000).
token_rate(2, 'gemma4-e2b', 'different', 0.900000).
boundary_token(2, 'identifies').
token_stat(2, 'identifies', 0.295807, 2.051271, 5).
token_rate(2, 'gemma4-e4b', 'identifies', 0.300000).
token_rate(2, 'gemma4-e2b', 'identifies', 0.900000).
boundary_token(2, 'information').
token_stat(2, 'information', 0.295807, 2.051271, 3).
token_rate(2, 'gemma4-e4b', 'information', 0.100000).
token_rate(2, 'gemma4-e2b', 'information', 0.700000).
boundary_token(2, 'information provided').
token_stat(2, 'information provided', 0.295807, 2.051271, 3).
token_rate(2, 'gemma4-e4b', 'information provided', 0.100000).
token_rate(2, 'gemma4-e2b', 'information provided', 0.700000).
boundary_token(2, 'm2 identifies').
token_stat(2, 'm2 identifies', 0.295807, 2.051271, 3).
token_rate(2, 'gemma4-e4b', 'm2 identifies', 0.100000).
token_rate(2, 'gemma4-e2b', 'm2 identifies', 0.700000).
boundary_token(2, 'option').
token_stat(2, 'option', 0.295807, 2.051271, 5).
token_rate(2, 'gemma4-e4b', 'option', 0.900000).
token_rate(2, 'gemma4-e2b', 'option', 0.300000).
boundary_token(2, 'option instances').
token_stat(2, 'option instances', 0.295807, 2.051271, 5).
token_rate(2, 'gemma4-e4b', 'option instances', 0.900000).
token_rate(2, 'gemma4-e2b', 'option instances', 0.300000).
boundary_token(2, 'significantly different').
token_stat(2, 'significantly different', 0.295807, 2.051271, 5).
token_rate(2, 'gemma4-e4b', 'significantly different', 0.300000).
token_rate(2, 'gemma4-e2b', 'significantly different', 0.900000).
boundary_token(2, 'signify').
token_stat(2, 'signify', 0.295807, 2.051271, 5).
token_rate(2, 'gemma4-e4b', 'signify', 0.300000).
token_rate(2, 'gemma4-e2b', 'signify', 0.900000).
dbn_unit(2, 1, 0.188722, ['option instances', 'option', 'chosen', 'chosen option', 'chosen chosen', 'leading', 'two two', 'leading two']).
dbn_unit(2, 2, 0.188722, ['significantly', 'frequency', 'm2 identifies', 'model', 'divergence', 'tokens', 'surprise', 'answers']).
dbn_unit(2, 3, 0.188722, ['divergence', 'surprise', 'tokens', 'answers', 'significantly', 'provided', 'meaningful surprise', 'based']).
