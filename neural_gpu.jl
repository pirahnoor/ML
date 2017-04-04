for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet,ArgParse,Compat,GZip

function generateData(upperbound,token)
	x=rand(1:9999999, upperbound,1);
	y=rand(1:9999999, upperbound,1);
	z=zeros(upperbound,1)
	if(isequal(token, "+"))
		z=x+y;
	elseif(isequal(token, "*"))
		z=x.*y;
	end
	s= Any[];
	zs= Any[];
	for i= 1: upperbound 
		a=bin(x[i])*token*bin(y[i]); 
		b=bin(z[i])
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
function init_state(E, input, vocab, w,m=24)
	c=1;k=1
	n=length(input);
	s=zeros(w,n,m)
	while c <= endof(input)
		ch=input[c];
		index= vocab[ch];
		v=E[index]
		s[1,k,:]=v;
		c=nextind(input,c)
		k=k+1
	end
	return s
	
end
function initWeight(kw,kh,x,y, vocab, atype=Array{Float32}, winit=0.1, m=24)
	w = Array(Any,8);
	#O=ones(m, length(vocab));
	w[1] = KnetArray{Float32}(winit*randn(kw,kh,m,m));#U
	w[2] = KnetArray{Float32}(zeros(1,1,m,1));#B
	w[3] = KnetArray{Float32}(winit*randn(kw,kh,m,m));#U'
	w[4] = KnetArray{Float32}(zeros(1,1,m,1));#B'
	w[5] = KnetArray{Float32}(winit*randn(kw,kh,m,m));#U''
	w[6] = KnetArray{Float32}(zeros(1,1,m,1));#B''
	w[7] = KnetArray{Float32}(randn(m, length(vocab))*winit);#W of softmax
	w[8] = Array{Float32}(zeros(1,length(vocab)));#b of softmax
	return w;
end
function CGRU(s,W)
	s=convert(KnetArray{Float32}, s);
	u=sigm(conv4(W[3], s, padding=1) .+ W[4]);
	r=sigm(conv4(W[5], s, padding=1) .+ W[6]);
	sn=u.*s+(1-u) .* tanh(conv4(W[1], (r.*s), padding=1).+W[2])
	return sn
end
function predict(s,w)
	sn=CGRU(s,w)
	y=Any[];
	for k=1:size(sn,2)
		lk=sn[1,k,:]'
		#lk=convert(KnetArray{Float32}, lk)
		lk=lk*w[7] .+ w[8];
		push!(y,lk);
	end
	return y
end
function loss(w, ygold, vocab, s, m=24)
	y=Any[];#4=vocab size
	total=0;count=0;
	c=1;k=1
	while c <= endof(ygold)#generating ygolds
		one_hot_vector= zeros(length(vocab), 1)
		ch=ygold[c];
		#println(ch)
		index= vocab[ch];
		one_hot_vector[index, 1]=1;
		one_hot_vector=convert(KnetArray{Float32}, one_hot_vector)
		push!(y,one_hot_vector);
		c=nextind(ygold,c)
		k=k+1
	end
	ypred=predict(s,w)
	for i=1: size(y,1)
		ynorm=logp(ypred[i])
		yg=y[i]
		total += sum(yg .* ynorm) 
	end
	return total/size(y,1)
	
end

function train(s, ygold, gclip, vocab, W)
	gloss=lossgradient(s, ygold[1], vocab, W)
	gloss2= gloss.*gloss
	gnorm=sqrt(sum(gloss2, 1))
	if gnorm >gclip
		gloss=(gloss*gclip)/gnorm
	end
	opts = map(x->Knet.Adam(), w);
	update!(sn,gloss, opts)

end
w=3;
m=24;
kh=3;
kw=3;
gclip=1
x, ygold=generateData(10000, "*");
data=minibatch(x, ygold, 1000)
E, vocab=embeddedMatrix();
s0=init_state(E, x[1], vocab,3 )
n=length(x[1]);
W=initWeight(kh,kw, w, n, vocab)
s0=reshape(s0,size(s0)..., 1)
sn=CGRU(s0,W)#upto here it is correct
lossgradient = grad(loss);
#train(sn, ygold, gclip,w)
l=lossgradient(W, ygold[1], vocab, s0 )
#size(y)




