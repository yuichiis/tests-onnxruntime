<?php

$filename = "onnxruntime.dll";
$headerFile = "onnxruntime.h";

$code = file_get_contents($headerFile);
$ffi = FFI::cdef($code,$filename);

echo get_class($ffi)."\n";
$version = ($ffi->OrtGetApiBase()[0]->GetVersionString)();
echo $version."\n";
$version_numbers = explode('.',$version);
[$major,$middle,$minor] = $version_numbers;
$api = ($ffi->OrtGetApiBase()[0]->GetApi)($middle)[0];
