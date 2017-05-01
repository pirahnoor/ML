for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet,ArgParse,Compat,GZip

function generateData(l,u, upperbound,token)
	x=rand(l:u, upperbound,1); #32 to 63   are binary numbers with length 6, 11 bits (1048, 1060) 20 bits(1000000,1000009)
	#y=x;
	y=rand(l:u, upperbound,1);
	z=zeros(upperbound,1)
	if(isequal(token, "+"))
		z=x+y;
	elseif(isequal(token, "*"))
		z=x.*y;
	elseif(isequal(token, "copy"))
		z=x;
	elseif(isequal(token, "reverse"))
		z=x;
	elseif(isequal(token, "duplicate"))
		z=x;
	elseif(isequal(token, "cbys"))
		z=x;
	end
	s= Any[];
	zs= Any[];
	for i= 1: upperbound 
		if(isequal(token, "+") || isequal(token, "*")) 
			a=bin(x[i])*token*bin(y[i]);
		else
			a=bin(x[i]);
		end
		if(isequal(token, "reverse"))
			b=reverse(bin(z[i]))
		elseif(isequal(token, "duplicate"))
			b=bin(z[i])*bin(z[i])
			xsz=length(bin(x[i]));
			for h=1:xsz
				a=a*"0"
			end
		elseif(isequal(token, "cbys"))
			count1=0; count0=0; c=1
			input=bin(z[i])
			while c <= endof(input)
				ch=input[c];
				#println(ch)
				if(isequal(ch, '1')) 
					count1=count1+1
				elseif(isequal(ch, '0'))
					count0= count0+1
				end
				c=nextind(input,c)
			end
			#println("0s= ",count0, " 1s=  ",count1)
			p=""
			q=""
			for k= 1:count0
			p=p*"0"
			end
			for l= 1:count1
			q=q*"1"
			end
			b=p*q
			#println(b, "---",p, "---",q)
		else
			b=bin(z[i])
		end
		push!(s,a) 
		push!(zs,b)                      
	end
	return s,zs
end
function minibatch(x, z, batchsize)
	data=Any[];
	for i=1:batchsize:size(x,1)-batchsize+1
		push!(data, x[i:i+batchsize-1], z[i:i+batchsize-1])
	end
	return data
	
end
function embeddedMatrix()
#We have four different characters 0, 1, + and *
	vocab = Dict{Char,Int}()
	vocab['0']=1
	vocab['1']=2
	vocab['+']=3
	vocab['*']=4
	return E=rand(4,24), vocab# m=24  
end
#input = string
function init_state(E, input, vocab, w=4,m=24)
	c=1;k=1
	n=length(input);
	#println(input)
	s=zeros(w,n,m)
	while c <= endof(input)
		ch=input[c];
		#println("ch -> ",ch)
		index= vocab[ch];
		#println("index -> ",index)
		v=E[index]
		s[1,k,:]=v;
		c=nextind(input,c)
		k=k+1
	end
	return s
	
end
function generate_ygold(ygold, vocab)
y=Any[];#4=vocab size
c=1;k=1
	while c <= endof(ygold)#generating ygolds
		one_hot_vector= KnetArray{Float32}(zeros(length(vocab), 1))
		ch=ygold[c];
		#println(ch)
		index= vocab[ch];
		one_hot_vector[index, 1]=1;
		#one_hot_vector=convert(KnetArray{Float32}, one_hot_vector)
		push!(y,one_hot_vector);
		c=nextind(ygold,c)
		k=k+1
	end
return y
end
function initWeight(kw,kh,x,y, vocab, atype=Array{Float32}, winit=0.001, m=24)
	w = Array(Any,8);
	#O=ones(m, length(vocab));

	w[1] = KnetArray{Float32}(xavier(kw,kh,m,m));#U, 
	w[2] = KnetArray{Float32}(zeros(1,1,m,1));#B
	w[3] = KnetArray{Float32}(xavier(kw,kh,m,m));#U'
	w[4] = KnetArray{Float32}(zeros(1,1,m,1));#B'
	w[5] = KnetArray{Float32}(xavier(kw,kh,m,m));#U''
	w[6] = KnetArray{Float32}(zeros(1,1,m,1));#B''
	w[7] = KnetArray{Float32}(xavier(m, length(vocab)));#W of softmax
	w[8] = KnetArray{Float32}(zeros(1,length(vocab)));#b of softmax
	return w;
end
function initparams(W)
    prms = map(x->Knet.Adam(lr=0.7 , beta1=0.95, beta2=0.995, eps=1e-4), W)
    return prms
end
function CGRU(s,W)
	u=sigm(conv4(W[3], s, padding=1) .+ W[4]);
	u=max(0,min(1,1.2*u-0.1)) #thresholding
	r=sigm(conv4(W[5], s, padding=1) .+ W[6]);
	r=max(0,min(1,1.2*r-0.1)) #thresholding
	sn=u.*s+(1-u) .* tanh(conv4(W[1], (r.*s), padding=1).+W[2])
	return sn
end
function predict(s,w, m=24)
	s=convert(KnetArray{Float32}, s);
	s1=CGRU(s,w)
	sn=CGRU(s1,w)
	
	y=Any[];
	for k=1:size(sn,2)
		#lk=KnetArray{Float32}(zeros(size(sn,3),1))
		lk=Any[]
		#generate lk form 4D sn array
		snew=reshape(sn, size(sn,1)*size(sn,2)*size(sn,3),1,1,1)
		for l=0:23
			lk= vcat(lk,snew[l*size(sn,1)*size(sn,2)+1])
		end
		lk=convert(KnetArray{Float32},lk)
		lk=reshape(lk,1,m)
		lk=relu(lk*w[7] .+ w[8]);
		push!(y,lk);
	end
	return y
end
function loss(w, ygold, vocab, s, m=24)
	total=0;count=0;
	ypred=predict(s,w)
	for i=1: size(ygold,1)
		ynorm=logp(ypred[i])
		yg=reshape(ygold[i], size(ynorm,1), size(ynorm,2))
		total +=(- sum(yg .* ynorm)) 
	end
	#println("LOSS Toatl", total)
	#divide by size(ygold,1) either
	return total/size(ygold,1)
	
end
lossgradient = grad(loss);

function train(x, ygold, gclip, W, E, vocab, prms)
	println("Training ...")
	for l= 1:size(ygold,1)
		#println(rand())
		ygoldn=generate_ygold(ygold[l], vocab)
		s=init_state(E, x[l], vocab )
		s=reshape(s, size(s)..., 1)
		gloss=lossgradient(W, ygoldn, vocab, s)
		gnorm=0;
		#print gradients to check
		#println(size(gloss[1]))
		#for k= 1 : size(gloss,1)
		#	println(gloss[k][1])
		#	println(gloss[k][2])
		#	println(gloss[k][3])
		#	println(gloss[k][4])
		#end
		for k= 1 : size(gloss,1)
			gnorm += sumabs2(gloss[k])
		end
		#println(gnorm)
		gnorm=sqrt(gnorm)
		if gnorm >gclip
			for k = 1: size(W,1)
				gloss[k] = (gloss[k] * gclip)/gnorm
				#gloss[k] = gloss[k]+rand_normal(l) #noise gradient
			end
		end
		#@show gradcheck(loss, W, ygoldn, vocab, s;verbose=true,atol=0.01 )
		for k = 1: size(W,1)
			update!(W[k], gloss[k], prms[k])
		end
	end
	return 
	
end
## return a random sample from a normal (Gaussian) distribution
function rand_normal(timestep, mean=0)
	eeta=0.01
	gamma=0.55
	variance = eeta/((1+timestep)^gamma)	
	stdev= sqrt(variance)
    if stdev <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand()
    u2 = rand()
    r = sqrt( -2.0*log(u1) )
    theta = 2.0*pi*u2
    return mean + stdev*r*sin(theta)
end
#ypred and ygold are arrays of one hot vectors of size Vocab X sequenceLength
#This method matches if the corresponding index of highest number in ypred[k] is equal to index of 1 in ygold[k]
function matchSequence(ypred, ygold)
	ncorrect=0;
	for i=1: size(ygold,1)
		#ypred 1X4 ygold 4X1
		ncorrect += sum(ygold[i] .* reshape(convert(KnetArray{Float32},(AutoGrad.getval(ypred)[i] .== maximum(AutoGrad.getval(ypred)[i],2))), size(ygold[i],1),size(ygold[i],2)))
		
	end
	#println("___________________________",size(ygold,1), "   ***    ", ncorrect);
	if ncorrect == size(ygold,1)
		return 1
	else
		return 0
	end
	
	
end
function accuracy(x,y,W, E, vocab)#send one x and one y from data
	println("Accuracy called")
	ncorrect =  0
	for i=1:size(y,1)
		ygold=generate_ygold(y[i], vocab)
		s0=init_state(E, x[i], vocab,3 )
		n=length(x[i]);
		s0=reshape(s0,size(s0)..., 1)
		ypred=predict(s0,W)
		ncorrect += matchSequence(ypred,ygold)
		#println("tic toc ", rand(), "  Correct ->", ncorrect)
	end

	
    return ncorrect/size(y,1)*100
end
#function main()
w=4;
m=24;
kh=3;
kw=3;
gclip=1
trainInstances= 1000
testInstances= 1000
lowerbound=0
upperbound=1000000
println("\n\n#######  NEURAL_GPU coded by Pirah Noor Soomro   #######\n\n ___________________________________________________________________")
println("Training Instances: ", trainInstances)
println("Testing Instances ", testInstances)
#=println("------------------MULTIPLY-----------------------")
x, ygold=generateData(trainInstances, "*");
E, vocab=embeddedMatrix();
n=length(x[1]);
W=initWeight(kh,kw, w, n, vocab)
prms = initparams(W)
train(x, ygold, gclip, W, E, vocab, prms)
println("Testing")
xtst, ygoldtst=generateData(lowerbound, 1606938000000000000000000000000000000000000000000000000000000, testInstances, "*"); #1606938000000000000000000000000000000000000000000000000000000 upto 200 bits
a=accuracy(xtst, ygoldtst,W, E, vocab)
println("Accuracy Binary MULTIPLY = ", a, "%.")=#
println("------------------ADD-----------------------")
x, ygold=generateData(trainInstances, "+");
E, vocab=embeddedMatrix();
n=length(x[1]);
W=initWeight(kh,kw, w, n, vocab)
prms = initparams(W)
train(x, ygold, gclip, W, E, vocab, prms)
println("Testing")
xtst, ygoldtst=generateData(lowerbound, 1606938000000000000000000000000000000000000000000000000000000, testInstances, "+"); #1606938000000000000000000000000000000000000000000000000000000 upto 200 bits
a=accuracy(xtst, ygoldtst,W, E, vocab)
println("Accuracy Binary ADD = ", a, "%.")
#=println("------------------COPY-----------------------")
x, ygold=generateData(trainInstances, "copy");
E, vocab=embeddedMatrix();
n=length(x[1]);
W=initWeight(kh,kw, w, n, vocab)
prms = initparams(W)
train(x, ygold, gclip, W, E, vocab, prms)
println("Testing")
xtst, ygoldtst=generateData(lowerbound, 1606938000000000000000000000000000000000000000000000000000000, testInstances, "copy"); #1606938000000000000000000000000000000000000000000000000000000 upto 200 bits
a=accuracy(xtst, ygoldtst,W, E, vocab)
println("Accuracy Binary COPY = ", a, "%.")
println("------------------REVERSE-----------------------")
x, ygold=generateData(trainInstances, "reverse");
E, vocab=embeddedMatrix();
n=length(x[1]);
W=initWeight(kh,kw, w, n, vocab)
prms = initparams(W)
train(x, ygold, gclip, W, E, vocab, prms)
println("Testing")
xtst, ygoldtst=generateData(lowerbound, 1606938000000000000000000000000000000000000000000000000000000, testInstances, "reverse"); #1606938000000000000000000000000000000000000000000000000000000 upto 200 bits
a=accuracy(xtst, ygoldtst,W, E, vocab)
println("Accuracy Binary REVERSE = ", a, "%.")
println("------------------DUPLICATE-----------------------")
x, ygold=generateData(trainInstances, "duplicate");
E, vocab=embeddedMatrix();
n=length(x[1]);
W=initWeight(kh,kw, w, n, vocab)
prms = initparams(W)
train(x, ygold, gclip, W, E, vocab, prms)
println("Testing")
xtst, ygoldtst=generateData(lowerbound, 1606938000000000000000000000000000000000000000000000000000000, testInstances, "duplicate"); #1606938000000000000000000000000000000000000000000000000000000 upto 200 bits
a=accuracy(xtst, ygoldtst,W, E, vocab)
println("Accuracy Binary DUPLICATE = ", a, "%.")
println("------------------Counting By Sorting-----------------------")
x, ygold=generateData(trainInstances, "cbys");
E, vocab=embeddedMatrix();
n=length(x[1]);
W=initWeight(kh,kw, w, n, vocab)
prms = initparams(W)
train(x, ygold, gclip, W, E, vocab, prms)
println("Testing")
xtst, ygoldtst=generateData(lowerbound, 1606938000000000000000000000000000000000000000000000000000000, testInstances, "cbys"); #1606938000000000000000000000000000000000000000000000000000000 upto 200 bits
a=accuracy(xtst, ygoldtst,W, E, vocab)
println("Accuracy Binary CByS = ", a, "%.") =#

#end
#main()



